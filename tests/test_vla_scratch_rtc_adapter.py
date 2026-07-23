from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataset_contracts.vla_scratch import (
    VLA_SCRATCH_ACTIONS_KEY,
    VLA_SCRATCH_IMAGE_MASKS_KEY,
    VLA_SCRATCH_IMAGES_KEY,
    VLA_SCRATCH_STATE_KEY,
    VLA_SCRATCH_TASK_KEY,
    VlaScratchProcessedSampleSpec,
    build_vla_scratch_processed_observation,
)
from deployments.client import FinalChunkRtcBackend, RtcModelRuntimeAdapter
from deployments.vla_scratch.client import (
    VlaScratchFinalChunkRtcConfig,
    create_vla_scratch_runtime_adapter,
)
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    NormalizedAction,
    PolicyInferenceRequest,
    nan_vector,
)
from robot_motion.deploy.inference import ASYNC_ALIGNED_CHUNK_MODE_V1
from robot_motion.deploy.observation_view import PolicyObservationView


HORIZON = 4
ACTION_DIM = 7
STATE_DIM = 7


class _CounterClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class _ExistingInferClientShape:
    """Same public shape as vla_scratch ZmqPolicyClient."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []
        self.cancel_count = 0

    def infer(self, observation: dict[str, Any]):
        self.requests.append(observation)
        return self.response

    def request_cancel(self) -> bool:
        self.cancel_count += 1
        return True


def _sample_spec() -> VlaScratchProcessedSampleSpec:
    return VlaScratchProcessedSampleSpec(
        source_revision="fa3cb37a6f3ba620bd2c167a97035b5431f31d19",
        camera_source_keys=("front", "right_wrist"),
        state_history=1,
        state_dim=STATE_DIM,
        action_horizon=HORIZON,
        action_dim=ACTION_DIM,
        state_representation_id="dexteleop_right_arm_state_unit_v1",
        action_representation_id="dexteleop_right_arm_absolute_pose_unit_v1",
    )


def _config() -> VlaScratchFinalChunkRtcConfig:
    return VlaScratchFinalChunkRtcConfig(
        sample_spec=_sample_spec(),
        provider_id="vla-scratch:unit",
        artifact_hash="artifact:vla-scratch-unit",
        action_codec_hash="codec:vla-scratch-unit",
        frame_contract_hash="frame:vla-scratch-unit",
        layout_hash="layout:vla-scratch-unit",
        normalization_hash="normalization:vla-scratch-unit",
        component_scope=("right_arm",),
        minimum_execution_horizon_steps=2,
    )


def _model_actions() -> np.ndarray:
    actions = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = np.arange(HORIZON, dtype=np.float32) * 0.01
    actions[:, 6] = 1.0
    return actions


def _decoder(model_actions, discrete_actions, request):
    del request
    assert discrete_actions is None
    decoded = []
    for row in model_actions:
        vector = nan_vector(DEPLOY_ACTION_DIM)
        vector[ACTION_ARM_POSE_SLICES["right"]] = row
        decoded.append(
            NormalizedAction(
                vector=vector,
                metadata={
                    "pose_action_source_mode": "absolute",
                    "pose_action_mode": "absolute",
                },
            )
        )
    return tuple(decoded)


def _actor_observation() -> PolicyObservationView:
    front = np.zeros((6, 8, 3), dtype=np.uint8)
    front[..., 0] = 10
    wrist = np.zeros((3, 6, 8), dtype=np.uint8)
    wrist[1, ...] = 20
    return PolicyObservationView(
        contract_id="contract:vla-scratch-unit",
        contract_hash="contract-hash:vla-scratch-unit",
        payload={
            "images": {"front": front, "right_wrist": wrist},
            "state": np.arange(STATE_DIM, dtype=np.float32),
        },
    )


def _request(adapter: RtcModelRuntimeAdapter) -> PolicyInferenceRequest:
    spec = adapter.request_spec
    return PolicyInferenceRequest(
        request_id="request:vla-scratch-unit",
        inference_id="inference:vla-scratch-unit",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id="session:vla-scratch-unit",
        control_domain_id="domain:vla-scratch-unit",
        producer_id="producer:vla-scratch-unit",
        producer_epoch=1,
        policy_node_id="node:vla-scratch-unit",
        artifact_hash=spec.artifact_hash,
        observation_cut_id="observation:vla-scratch-unit",
        observation_timestamp=9.9,
        observation_step_index=99,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.1,
        clock_domain_id="monotonic:unit",
        action_period_sec=0.1,
        prediction_horizon_steps=spec.prediction_horizon_steps,
        minimum_execution_horizon_steps=spec.minimum_execution_horizon_steps,
        source_plan_id="plan:vla-scratch-unit",
        source_plan_generation=0,
        source_plan_hash="plan-hash:vla-scratch-unit",
        source_plan_cursor_at=10.0,
        producer_plan_committed_through=None,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay:vla-scratch-unit",
        action_codec_hash=spec.action_codec_hash,
        frame_contract_hash=spec.frame_contract_hash,
        layout_hash=spec.layout_hash,
        normalization_hash=spec.normalization_hash,
        valid_until=11.0,
    )


def test_processed_observation_matches_existing_serving_keys_and_order() -> None:
    spec = _sample_spec()
    front = np.zeros((5, 7, 3), dtype=np.uint8)
    front[..., 0] = 11
    wrist = np.zeros((3, 5, 7), dtype=np.uint8)
    wrist[1, ...] = 22
    observation = build_vla_scratch_processed_observation(
        spec=spec,
        images_by_source={"right_wrist": wrist, "front": front},
        state=np.arange(STATE_DIM, dtype=np.float32),
        task="pick the object",
    )
    payload = observation.to_wire_payload()

    assert tuple(payload) == (
        VLA_SCRATCH_IMAGES_KEY,
        VLA_SCRATCH_IMAGE_MASKS_KEY,
        VLA_SCRATCH_STATE_KEY,
        VLA_SCRATCH_TASK_KEY,
    )
    assert payload[VLA_SCRATCH_IMAGES_KEY].shape == (2, 3, 5, 7)
    assert payload[VLA_SCRATCH_IMAGES_KEY].dtype == np.uint8
    assert payload[VLA_SCRATCH_IMAGE_MASKS_KEY].shape == (2, 1)
    assert payload[VLA_SCRATCH_IMAGE_MASKS_KEY].dtype == np.bool_
    assert np.all(payload[VLA_SCRATCH_IMAGES_KEY][0, 0] == 11)
    assert np.all(payload[VLA_SCRATCH_IMAGES_KEY][1, 1] == 22)
    assert payload[VLA_SCRATCH_STATE_KEY].shape == (1, STATE_DIM)
    assert payload[VLA_SCRATCH_TASK_KEY] == "pick the object"

    payload[VLA_SCRATCH_IMAGES_KEY][...] = 255
    assert not np.all(observation.images == 255)


def test_processed_observation_fails_closed_on_missing_camera_or_state_shape() -> None:
    spec = _sample_spec()
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Missing required"):
        build_vla_scratch_processed_observation(
            spec=spec,
            images_by_source={"front": image},
            state=np.zeros(STATE_DIM, dtype=np.float32),
            task="unit task",
        )
    with pytest.raises(ValueError, match="state must have shape"):
        build_vla_scratch_processed_observation(
            spec=spec,
            images_by_source={"front": image, "right_wrist": image},
            state=np.zeros(STATE_DIM + 1, dtype=np.float32),
            task="unit task",
        )


def test_existing_infer_client_shape_composes_without_framework_subclass() -> None:
    transport = _ExistingInferClientShape(
        {
            VLA_SCRATCH_ACTIONS_KEY: _model_actions(),
            "server_timing": {"infer_s": 0.01},
        }
    )
    adapter = create_vla_scratch_runtime_adapter(
        transport=transport,
        config=_config(),
        decoder=_decoder,
        task="unit task",
        clock=_CounterClock(),
    )
    assert isinstance(adapter, RtcModelRuntimeAdapter)
    assert isinstance(adapter.backend, FinalChunkRtcBackend)

    bundles = []
    result = adapter.create_worker_executor(observe_bundle=bundles.append)(
        _request(adapter),
        _actor_observation(),
    )

    assert result.request_id == "request:vla-scratch-unit"
    assert result.source_chunk_size == HORIZON
    assert len(result.decoded_absolute_actions) == HORIZON
    assert len(bundles) == 1
    assert bundles[0].response_validation.accepted
    assert bundles[0].conditioning_receipt is None
    assert len(transport.requests) == 1
    assert set(transport.requests[0]) == {
        VLA_SCRATCH_IMAGES_KEY,
        VLA_SCRATCH_IMAGE_MASKS_KEY,
        VLA_SCRATCH_STATE_KEY,
        VLA_SCRATCH_TASK_KEY,
    }
    np.testing.assert_array_equal(
        result.decoded_absolute_actions[2].vector[
            ACTION_ARM_POSE_SLICES["right"]
        ],
        _model_actions()[2],
    )

    executor = adapter.create_worker_executor()
    assert executor.request_cancel()
    assert transport.cancel_count == 1


@pytest.mark.parametrize(
    ("response", "error"),
    (
        ({}, "missing exact output key"),
        (
            {VLA_SCRATCH_ACTIONS_KEY: np.zeros((HORIZON - 1, ACTION_DIM))},
            "actions must have shape",
        ),
        (
            {
                VLA_SCRATCH_ACTIONS_KEY: np.full(
                    (HORIZON, ACTION_DIM),
                    np.nan,
                    dtype=np.float32,
                )
            },
            "actions must be finite",
        ),
        (
            {
                VLA_SCRATCH_ACTIONS_KEY: np.zeros(
                    (HORIZON, ACTION_DIM), dtype=np.int64
                )
            },
            "floating dtype",
        ),
    ),
)
def test_response_contract_fails_closed(response: Any, error: str) -> None:
    adapter = create_vla_scratch_runtime_adapter(
        transport=_ExistingInferClientShape(response),
        config=_config(),
        decoder=_decoder,
        task="unit task",
        clock=_CounterClock(),
    )
    with pytest.raises((TypeError, ValueError), match=error):
        adapter.create_worker_executor()(_request(adapter), _actor_observation())


def test_vla_scratch_addon_does_not_import_baseline_or_runtime_owners() -> None:
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        project_root / "dataset_contracts/vla_scratch.py",
        project_root / "deployments/client/rtc_final_chunk.py",
        project_root / "deployments/vla_scratch/client/rtc.py",
    )
    forbidden = (
        "vla_scratch",
        "control_station",
        "teleop_runtime",
        "sim",
        "hardware",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            value == prefix or value.startswith(prefix + ".")
            for value in imports
            for prefix in forbidden
        ), (path, imports)
