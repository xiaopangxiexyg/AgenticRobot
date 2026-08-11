from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts import (
    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1,
    PolicyArmHoldOutputContract,
    PolicyArtifactContract,
)
from dataset_contracts.v22 import (
    DatasetLayoutBindingError,
    EndEffectorSpec,
    PolicyIOLayout,
)
from deployments.client import (
    RemoteModelPolicy,
    RemotePolicyConfig,
    ServerMetadataError,
    UnifiedActionChunk,
    deploy_contract_from_policy_artifact,
    validate_server_metadata,
)
from deployments.openpi.client import OpenPISimAdapterConfig
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    DEPLOY_ACTION_DIM,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    ArmHoldKeyframePrediction,
    ArmHoldResolvedHistoryBinding,
    ArmHoldSemanticResolver,
    CanonicalDeployActionPipeline,
    NormalizedAction,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    PolicyActionEmissionResult,
    PolicyAdapterConfig,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
    V22DeploymentAdapter,
    build_deployment_observation,
    nan_vector,
)
from robot_motion.deploy.action_semantics import bind_arm_hold_predictions


def _contract(
    *,
    enter_steps: int = 1,
    exit_steps: int = 1,
) -> PolicyArmHoldOutputContract:
    return PolicyArmHoldOutputContract(
        output_key="arm_hold_probability",
        training_contract_hash=(
            DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
        ),
        enter_threshold=0.8,
        exit_threshold=0.2,
        enter_consecutive_steps=enter_steps,
        exit_consecutive_steps=exit_steps,
    )


def _pose(x: float) -> np.ndarray:
    return np.asarray(
        [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )


def _observation(
    *,
    left_x: float = -0.1,
    right_x: float = 0.1,
    left_hand: float = 0.25,
    right_hand: float = 0.75,
    step_index: int = 1,
):
    left_pose = np.eye(4, dtype=float)
    left_pose[:3, 3] = [left_x, 0.0, 0.3]
    right_pose = np.eye(4, dtype=float)
    right_pose[:3, 3] = [right_x, 0.0, 0.3]
    last_action = nan_vector(DEPLOY_ACTION_DIM)
    last_action[ACTION_ARM_POSE_SLICES["left"]] = _pose(left_x)
    last_action[ACTION_ARM_POSE_SLICES["right"]] = _pose(right_x)
    last_action[ACTION_HAND_SLICES["left"]] = left_hand
    last_action[ACTION_HAND_SLICES["right"]] = right_hand
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        step_index=step_index,
        arm_states={
            "left": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": left_pose,
            },
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": right_pose,
            },
        },
        hand_states={
            "left": {"joint_pos": np.full(6, left_hand, dtype=np.float32)},
            "right": {"joint_pos": np.full(6, right_hand, dtype=np.float32)},
        },
        last_action=last_action,
        timestamp=float(step_index),
    )


def _action(
    contract: PolicyArmHoldOutputContract,
    probabilities: tuple[float, float],
    *,
    left_x: float = -0.4,
    right_x: float = 0.4,
    left_hand: float = 0.9,
    right_hand: float = 0.1,
) -> NormalizedAction:
    vector = nan_vector(DEPLOY_ACTION_DIM)
    vector[ACTION_ARM_POSE_SLICES["left"]] = _pose(left_x)
    vector[ACTION_ARM_POSE_SLICES["right"]] = _pose(right_x)
    vector[ACTION_HAND_SLICES["left"]] = left_hand
    vector[ACTION_HAND_SLICES["right"]] = right_hand
    return NormalizedAction(
        vector=vector,
        arm_hold_prediction=ArmHoldKeyframePrediction(
            probabilities=probabilities,
            contract_hash=contract.contract_hash,
        ),
    )


@pytest.mark.parametrize(
    ("probabilities", "combination"),
    (
        ((0.0, 0.0), "00"),
        ((1.0, 0.0), "10"),
        ((0.0, 1.0), "01"),
        ((1.0, 1.0), "11"),
    ),
)
def test_resolver_supports_all_four_independent_side_combinations(
    probabilities: tuple[float, float],
    combination: str,
) -> None:
    contract = _contract()
    observation = _observation()
    model_action = _action(contract, probabilities)
    resolver = ArmHoldSemanticResolver(
        contract,
        active_sides=("left", "right"),
    )

    resolved, receipt = resolver.resolve(model_action, observation)

    assert receipt.source_combination == combination
    assert receipt.resolved_combination == combination
    for index, side in enumerate(("left", "right")):
        expected = (
            observation.hold_action()
            if combination[index] == "1"
            else model_action.vector
        )
        np.testing.assert_allclose(
            resolved.vector[ACTION_ARM_POSE_SLICES[side]],
            expected[ACTION_ARM_POSE_SLICES[side]],
        )
        np.testing.assert_allclose(
            resolved.vector[ACTION_HAND_SLICES[side]],
            expected[ACTION_HAND_SLICES[side]],
        )


def test_hold_reference_is_latched_until_hysteretic_exit() -> None:
    contract = _contract()
    resolver = ArmHoldSemanticResolver(
        contract,
        active_sides=("left", "right"),
    )
    entry_observation = _observation(left_x=-0.1, right_x=0.1)
    entered, _ = resolver.resolve(
        _action(contract, (1.0, 1.0)),
        entry_observation,
    )
    drifted_observation = _observation(
        left_x=-0.2,
        right_x=0.2,
        step_index=2,
    )

    held, held_receipt = resolver.resolve(
        _action(contract, (0.5, 0.5), left_x=-0.5, right_x=0.5),
        drifted_observation,
    )
    exited, exited_receipt = resolver.resolve(
        _action(contract, (0.0, 0.0), left_x=-0.6, right_x=0.6),
        drifted_observation,
    )

    np.testing.assert_allclose(held.vector, entered.vector)
    assert held_receipt.resolved_combination == "11"
    assert exited_receipt.resolved_combination == "00"
    assert exited.vector[ACTION_ARM_POSE_SLICES["left"]][0] == pytest.approx(
        -0.6
    )
    assert exited.vector[ACTION_ARM_POSE_SLICES["right"]][0] == pytest.approx(
        0.6
    )


def test_enter_and_exit_debounce_counts_are_explicit() -> None:
    contract = _contract(enter_steps=2, exit_steps=2)
    resolver = ArmHoldSemanticResolver(
        contract,
        active_sides=("left", "right"),
    )
    observation = _observation()

    _, first_enter = resolver.resolve(
        _action(contract, (1.0, 0.0)),
        observation,
    )
    _, second_enter = resolver.resolve(
        _action(contract, (1.0, 0.0)),
        observation,
    )
    _, first_exit = resolver.resolve(
        _action(contract, (0.0, 0.0)),
        observation,
    )
    _, second_exit = resolver.resolve(
        _action(contract, (0.0, 0.0)),
        observation,
    )

    assert first_enter.resolved_combination == "00"
    assert first_enter.side_resolutions[0].consecutive_count == 1
    assert second_enter.resolved_combination == "10"
    assert second_enter.side_resolutions[0].consecutive_count == 2
    assert first_exit.resolved_combination == "10"
    assert first_exit.side_resolutions[0].consecutive_count == 1
    assert second_exit.resolved_combination == "00"
    assert second_exit.side_resolutions[0].consecutive_count == 2


def test_artifact_contract_is_additive_and_server_handshake_is_strict() -> None:
    contract = _contract()
    layout = PolicyIOLayout(
        name="arm_hold_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )
    legacy = PolicyArtifactContract(policy_io_layout=layout)
    assert "arm_hold_output_contract" not in legacy.to_dict()

    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        arm_hold_output=_contract(),
    )
    payload = artifact.to_dict()
    restored = PolicyArtifactContract.from_dict(payload)
    assert restored.to_dict() == payload
    assert len(restored.outputs) == 2

    deploy_contract = deploy_contract_from_policy_artifact(restored)
    metadata = {
        "deploy_schema_version": artifact.deploy_schema_version,
        "local_robot": {
            "input_state_key": artifact.state_key,
            "input_state_discrete_key": artifact.state_discrete_key,
            "output_action_dim": layout.action_dim,
            "output_action_discrete_dim": layout.action_discrete_dim,
            "input_state_dim": layout.state_dim,
            "input_state_discrete_dim": layout.state_discrete_dim,
            "active_sides": ["left", "right"],
        },
        "dataset_adapter": {
            "policy_io_layout_hash": layout.layout_hash,
        },
        "action_semantics": {
            "arm_hold_output_key": _contract().output_key,
            "arm_hold_output_contract_hash": _contract().contract_hash,
        },
    }
    assert validate_server_metadata(metadata, deploy_contract).ok
    for semantic_metadata, expected_error in (
        (
            {
                "arm_hold_output_key": "wrong",
                "arm_hold_output_contract_hash": contract.contract_hash,
            },
            "arm_hold_output_key_mismatch",
        ),
        (
            {
                "arm_hold_output_key": contract.output_key,
                "arm_hold_output_contract_hash": "wrong",
            },
            "arm_hold_output_contract_hash_mismatch",
        ),
    ):
        bad_metadata = {
            **metadata,
            "action_semantics": semantic_metadata,
        }
        with pytest.raises(ServerMetadataError, match=expected_error):
            validate_server_metadata(bad_metadata, deploy_contract)
    del metadata["action_semantics"]
    with pytest.raises(ServerMetadataError, match="arm_hold_output"):
        validate_server_metadata(metadata, deploy_contract)
    with pytest.raises(DatasetLayoutBindingError, match="must not collide"):
        PolicyArtifactContract(
            policy_io_layout=layout,
            arm_hold_output=PolicyArmHoldOutputContract(
                output_key="actions",
                training_contract_hash=(
                    DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
                ),
                enter_threshold=0.8,
                exit_threshold=0.2,
                enter_consecutive_steps=1,
                exit_consecutive_steps=1,
            ),
        )


def test_artifact_rejects_orphan_or_tampered_arm_hold_contract_hash() -> None:
    layout = PolicyIOLayout.legacy_v21_q6()
    legacy_payload = PolicyArtifactContract(
        policy_io_layout=layout,
    ).to_dict()
    orphan = {
        **legacy_payload,
        "arm_hold_output_contract_hash": _contract().contract_hash,
    }
    with pytest.raises(DatasetLayoutBindingError, match="hash without"):
        PolicyArtifactContract.from_dict(orphan)

    artifact_payload = PolicyArtifactContract(
        policy_io_layout=layout,
        arm_hold_output=_contract(),
    ).to_dict()
    artifact_payload["arm_hold_output_contract_hash"] = "wrong"
    with pytest.raises(DatasetLayoutBindingError, match="does not match"):
        PolicyArtifactContract.from_dict(artifact_payload)


class _Policy:
    name = "arm-hold-test"

    def __init__(self, output):
        self.output = output

    def predict(self, _observation):
        return self.output


class _Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
                end_effector_sides=("left", "right"),
                max_arm_xyz_step_m=1.0,
                max_arm_rot_step_deg=180.0,
                max_arm_joint_step_rad=1.0,
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    )


def test_sync_runtime_pairs_scores_with_actions_and_resolves_only_on_pop() -> None:
    contract = _contract()
    observation = _observation()
    model = _action(contract, (1.0, 0.0))
    output = {
        "actions": np.stack((model.vector, model.vector)),
        "action_semantics": {
            contract.output_key: np.asarray(
                ((1.0, 0.0), (0.0, 1.0)),
                dtype=np.float32,
            ),
            "contract_hash": contract.contract_hash,
        },
    }
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            arm_hold_output=contract,
        ),
        monotonic_clock=_Clock(),
    )

    prediction = runtime.predict_and_queue(
        _Policy(output),
        observation,
        now=10.0,
    )
    assert prediction.accepted is True
    assert runtime._arm_hold_resolver.resolved_combination == "00"

    first = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="test",
    )

    assert first.emission_error is None
    assert first.arm_hold_resolution is not None
    assert first.arm_hold_resolution.resolved_combination == "10"
    assert first.emission is not None
    assert first.emission.model_decoded_absolute_action is not None
    assert first.emission.arm_hold_resolution is first.arm_hold_resolution
    assert first.raw_action is not None
    assert first.prepared_action is not None
    assert first.raw_lookahead_action is None
    assert runtime.action_queue_length() == 1
    np.testing.assert_allclose(
        first.raw_action.vector[ACTION_ARM_POSE_SLICES["left"]],
        observation.hold_action()[ACTION_ARM_POSE_SLICES["left"]],
    )
    np.testing.assert_allclose(
        first.raw_action.vector[ACTION_ARM_POSE_SLICES["right"]],
        model.vector[ACTION_ARM_POSE_SLICES["right"]],
    )
    np.testing.assert_allclose(
        first.prepared_action.action.vector[ACTION_ARM_POSE_SLICES["left"]],
        observation.hold_action()[ACTION_ARM_POSE_SLICES["left"]],
    )

    second = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="test",
    )
    assert second.emission_error is None
    assert second.arm_hold_resolution is not None
    assert second.arm_hold_resolution.resolved_combination == "01"
    assert runtime.action_queue_length() == 0


def test_declared_semantics_missing_or_misaligned_rejects_whole_chunk() -> None:
    contract = _contract()
    observation = _observation()
    action = _action(contract, (0.0, 0.0)).vector
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            arm_hold_output=contract,
        ),
        monotonic_clock=_Clock(),
    )

    missing = runtime.predict_and_queue(
        _Policy({"actions": np.stack((action, action))}),
        observation,
        now=10.0,
    )
    assert missing.error is not None
    assert runtime.action_queue_length() == 0

    misaligned = runtime.predict_and_queue(
        _Policy(
            {
                "actions": np.stack((action, action)),
                "action_semantics": {
                    contract.output_key: np.asarray(((0.0, 0.0),)),
                    "contract_hash": contract.contract_hash,
                },
            }
        ),
        observation,
        now=11.0,
    )
    assert misaligned.error is not None
    assert runtime.action_queue_length() == 0


def test_legacy_runtime_ignores_undeclared_semantics_without_action_change() -> None:
    observation = _observation()
    contract = _contract()
    action = _action(contract, (1.0, 1.0)).vector
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        ),
        monotonic_clock=_Clock(),
    )
    output = {
        "actions": action,
        "action_semantics": {
            contract.output_key: np.asarray((1.0, 1.0)),
            "contract_hash": contract.contract_hash,
        },
    }

    result = runtime.predict_and_queue(_Policy(output), observation, now=10.0)

    assert result.accepted is True
    np.testing.assert_array_equal(result.actions[0].vector, action)
    assert result.actions[0].arm_hold_prediction is None


class _Transport:
    def __init__(self, *, metadata, response) -> None:
        self.metadata = metadata
        self.response = response

    def infer(self, _observation):
        return self.response

    def reset(self) -> None:
        return None


def test_remote_client_and_v22_bridge_preserve_the_typed_sidecar() -> None:
    contract = _contract()
    layout = PolicyIOLayout(
        name="arm_hold_remote_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        arm_hold_output=contract,
    )
    deploy_contract = deploy_contract_from_policy_artifact(artifact)
    metadata = {
        "deploy_schema_version": artifact.deploy_schema_version,
        "local_robot": {
            "input_state_key": artifact.state_key,
            "input_state_discrete_key": artifact.state_discrete_key,
            "input_state_dim": layout.state_dim,
            "input_state_discrete_dim": layout.state_discrete_dim,
            "output_action_dim": layout.action_dim,
            "output_action_discrete_dim": layout.action_discrete_dim,
            "active_sides": ["left", "right"],
        },
        "dataset_adapter": {"policy_io_layout_hash": layout.layout_hash},
        "action_semantics": {
            "arm_hold_output_key": contract.output_key,
            "arm_hold_output_contract_hash": contract.contract_hash,
        },
    }
    response = {
        artifact.action_key: np.zeros((2, layout.action_dim), dtype=np.float32),
        contract.output_key: np.asarray(
            ((1.0, 0.0), (0.0, 1.0)),
            dtype=np.float32,
        ),
        "arm_hold_output_contract_hash": contract.contract_hash,
    }
    policy = RemoteModelPolicy(
        transport=_Transport(metadata=metadata, response=response),
        observation_adapter=lambda _value: {},
        config=RemotePolicyConfig(
            action_key=artifact.action_key,
            action_dim=layout.action_dim,
            action_discrete_dim=0,
            contract=deploy_contract,
            arm_hold_output=contract,
        ),
    )

    chunk = policy.predict(object())
    normalized = V22DeploymentAdapter(layout).to_legacy_normalized_actions(
        chunk
    )

    assert [action.arm_hold_prediction.probabilities for action in normalized] == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]
    assert all(
        action.arm_hold_prediction.contract_hash == contract.contract_hash
        for action in normalized
    )


def _contract_payload() -> dict[str, object]:
    return _contract().to_dict()


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema", "future", "Unsupported"),
        ("output_key", " ", "non-empty"),
        ("training_contract_hash", "0" * 64, "stable"),
        ("enter_threshold", np.nan, r"\[0, 1\]"),
        ("exit_threshold", np.inf, r"\[0, 1\]"),
        ("enter_threshold", True, "finite probability"),
        ("enter_threshold", 0.2, "greater than"),
        ("exit_threshold", 0.8, "greater than"),
        ("enter_consecutive_steps", 0, "positive integer"),
        ("exit_consecutive_steps", -1, "positive integer"),
        ("enter_consecutive_steps", 1.5, "positive integer"),
        ("exit_consecutive_steps", True, "positive integer"),
        ("side_order", ("right", "left"), "side_order"),
        ("score_semantics", "logit", "probabilities"),
        ("target_alignment", "chunk_start", "action keyframes"),
        ("positive_scope", "arm_only", "arm_and_end_effector"),
        ("supported_combinations", ("00", "11"), "00, 01, 10, and 11"),
    ),
)
def test_output_contract_rejects_ambiguous_or_unsupported_semantics(
    field: str,
    value: object,
    error: str,
) -> None:
    payload = _contract_payload()
    payload[field] = value

    with pytest.raises(DatasetLayoutBindingError, match=error):
        PolicyArmHoldOutputContract.from_dict(payload)


def test_output_contract_roundtrip_requires_fields_and_authenticates_hash() -> None:
    payload = _contract().to_dict(include_hash=True)
    assert PolicyArmHoldOutputContract.from_dict(payload) == _contract()

    missing = dict(payload)
    del missing["exit_threshold"]
    with pytest.raises(DatasetLayoutBindingError, match="missing required"):
        PolicyArmHoldOutputContract.from_dict(missing)

    tampered = dict(payload)
    tampered["contract_hash"] = "sha256:" + "0" * 64
    with pytest.raises(DatasetLayoutBindingError, match="does not match"):
        PolicyArmHoldOutputContract.from_dict(tampered)


@pytest.mark.parametrize(
    "probabilities",
    (
        (0.5,),
        (0.5, 0.5, 0.5),
        (-0.1, 0.5),
        (1.1, 0.5),
        (np.nan, 0.5),
        (np.inf, 0.5),
    ),
)
def test_keyframe_prediction_rejects_invalid_probability_pairs(
    probabilities: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="two finite values"):
        ArmHoldKeyframePrediction(
            probabilities=probabilities,  # type: ignore[arg-type]
            contract_hash=_contract().contract_hash,
        )

    with pytest.raises(ValueError, match="non-empty"):
        ArmHoldKeyframePrediction(
            probabilities=(0.0, 0.0),
            contract_hash=" ",
        )


@pytest.mark.parametrize(
    ("probabilities", "contract_hash", "error"),
    (
        (None, "hash", "supplied together"),
        (np.zeros((1, 2), dtype=np.float32), None, "supplied together"),
        (np.zeros((2, 2), dtype=np.float32), "hash", "shape"),
        (np.zeros((1, 3), dtype=np.float32), "hash", "shape"),
        (np.asarray(((np.nan, 0.0),)), "hash", r"\[0, 1\]"),
        (np.asarray(((-0.01, 0.0),)), "hash", r"\[0, 1\]"),
        (np.asarray(((1.01, 0.0),)), "hash", r"\[0, 1\]"),
        (np.zeros((1, 2), dtype=np.float32), " ", "non-empty"),
    ),
)
def test_unified_chunk_rejects_unpaired_misaligned_or_invalid_sidecar(
    probabilities: object,
    contract_hash: str | None,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        UnifiedActionChunk(
            actions=np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32),
            arm_hold_probabilities=probabilities,
            arm_hold_contract_hash=contract_hash,
        )


@pytest.mark.parametrize(
    "probabilities",
    (
        np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32),
        np.asarray(((np.nan, 0.0), (0.0, 0.0)), dtype=np.float32),
        np.asarray(((-0.1, 0.0), (0.0, 0.0)), dtype=np.float32),
        np.asarray(((1.1, 0.0), (0.0, 0.0)), dtype=np.float32),
    ),
)
def test_binding_rejects_shape_nonfinite_and_logits_atomically(
    probabilities: np.ndarray,
) -> None:
    contract = _contract()
    actions = [
        NormalizedAction(vector=np.zeros(DEPLOY_ACTION_DIM, dtype=np.float32)),
        NormalizedAction(vector=np.zeros(DEPLOY_ACTION_DIM, dtype=np.float32)),
    ]
    output = {
        "action_semantics": {
            contract.output_key: probabilities,
            "contract_hash": contract.contract_hash,
        }
    }

    with pytest.raises(ValueError):
        bind_arm_hold_predictions(output, actions, contract=contract)
    assert all(action.arm_hold_prediction is None for action in actions)


def test_binding_rejects_partial_embedded_predictions_and_wrong_hash() -> None:
    contract = _contract()
    plain = NormalizedAction(vector=np.zeros(DEPLOY_ACTION_DIM, dtype=np.float32))
    predicted = _action(contract, (0.0, 0.0))

    with pytest.raises(ValueError, match="every action keyframe"):
        bind_arm_hold_predictions({}, (predicted, plain), contract=contract)

    wrong = replace(
        predicted,
        arm_hold_prediction=ArmHoldKeyframePrediction(
            probabilities=(0.0, 0.0),
            contract_hash="wrong",
        ),
    )
    with pytest.raises(ValueError, match="contract hash mismatch"):
        bind_arm_hold_predictions({}, (wrong,), contract=contract)


def test_threshold_boundaries_are_inclusive_and_hysteresis_band_is_sticky() -> None:
    contract = _contract()
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    observation = _observation()

    _, entered = resolver.resolve(
        _action(contract, (contract.enter_threshold, 0.0)),
        observation,
    )
    _, sticky = resolver.resolve(
        _action(contract, (0.5, 0.0)),
        observation,
    )
    _, exited = resolver.resolve(
        _action(contract, (contract.exit_threshold, 0.0)),
        observation,
    )

    assert entered.resolved_combination == "10"
    assert sticky.resolved_combination == "10"
    assert exited.resolved_combination == "00"


def test_debounce_counters_reset_when_evidence_is_interrupted() -> None:
    contract = _contract(enter_steps=2, exit_steps=2)
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    observation = _observation()

    resolver.resolve(_action(contract, (1.0, 0.0)), observation)
    _, enter_reset = resolver.resolve(_action(contract, (0.79, 0.0)), observation)
    _, enter_again = resolver.resolve(_action(contract, (1.0, 0.0)), observation)
    _, entered = resolver.resolve(_action(contract, (1.0, 0.0)), observation)
    resolver.resolve(_action(contract, (0.0, 0.0)), observation)
    _, exit_reset = resolver.resolve(_action(contract, (0.21, 0.0)), observation)
    _, exit_again = resolver.resolve(_action(contract, (0.0, 0.0)), observation)
    _, exited = resolver.resolve(_action(contract, (0.0, 0.0)), observation)

    assert enter_reset.side_resolutions[0].consecutive_count == 0
    assert enter_again.resolved_combination == "00"
    assert entered.resolved_combination == "10"
    assert exit_reset.side_resolutions[0].consecutive_count == 0
    assert exit_again.resolved_combination == "10"
    assert exited.resolved_combination == "00"


def test_inactive_side_prediction_never_modifies_or_latches_that_side() -> None:
    contract = _contract()
    observation = _observation()
    model = _action(contract, (1.0, 1.0))
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left",))

    resolved, receipt = resolver.resolve(model, observation)

    assert receipt.source_combination == "11"
    assert receipt.resolved_combination == "10"
    assert receipt.side_resolutions[1].transition == "inactive"
    np.testing.assert_array_equal(
        resolved.vector[ACTION_ARM_POSE_SLICES["right"]],
        model.vector[ACTION_ARM_POSE_SLICES["right"]],
    )
    np.testing.assert_array_equal(
        resolved.vector[ACTION_HAND_SLICES["right"]],
        model.vector[ACTION_HAND_SLICES["right"]],
    )


def test_hold_prefers_last_commanded_end_effector_reference_over_measurement() -> None:
    contract = _contract()
    observation = _observation(left_hand=0.25)
    last_action = observation.last_action.copy()
    last_action[ACTION_HAND_SLICES["left"]] = 0.9
    observation = replace(observation, last_action=last_action)
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    resolved, _ = resolver.resolve(_action(contract, (1.0, 0.0)), observation)

    np.testing.assert_allclose(
        resolved.vector[ACTION_HAND_SLICES["left"]],
        0.9,
    )


def test_hold_falls_back_to_measured_end_effector_when_last_action_is_missing() -> None:
    contract = _contract()
    observation = _observation(left_hand=0.35)
    last_action = observation.last_action.copy()
    last_action[ACTION_HAND_SLICES["left"]] = np.nan
    observation = replace(observation, last_action=last_action)
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    resolved, _ = resolver.resolve(_action(contract, (1.0, 0.0)), observation)

    np.testing.assert_allclose(
        resolved.vector[ACTION_HAND_SLICES["left"]],
        0.35,
    )


def test_two_side_entry_is_atomic_when_second_reference_is_missing() -> None:
    contract = _contract()
    observation = _observation()
    state = observation.state.copy()
    last_action = observation.last_action.copy()
    state[STATE_ARM_POSE_SLICES["right"]] = np.nan
    last_action[ACTION_ARM_POSE_SLICES["right"]] = np.nan
    missing_right = replace(
        observation,
        state=state,
        last_action=last_action,
    )
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    with pytest.raises(ValueError, match="right arm hold reference"):
        resolver.resolve(_action(contract, (1.0, 1.0)), missing_right)

    assert resolver.resolved_combination == "00"
    _, receipt = resolver.resolve(
        _action(contract, (0.5, 0.0)),
        observation,
    )
    assert receipt.resolved_combination == "00"


def test_entry_fails_closed_when_end_effector_reference_is_missing() -> None:
    contract = _contract()
    observation = _observation()
    state = observation.state.copy()
    last_action = observation.last_action.copy()
    state[STATE_HAND_SLICES["left"]] = np.nan
    last_action[ACTION_HAND_SLICES["left"]] = np.nan
    missing_hand = replace(observation, state=state, last_action=last_action)
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    with pytest.raises(ValueError, match="end-effector hold reference"):
        resolver.resolve(_action(contract, (1.0, 0.0)), missing_hand)

    assert resolver.resolved_combination == "00"


def test_joint_target_hold_latches_measured_qpos_not_model_target() -> None:
    contract = _contract()
    observation = _observation()
    state = observation.state.copy()
    measured = np.linspace(0.1, 0.6, 6, dtype=np.float32)
    state[STATE_ARM_QPOS_SLICES["left"]] = measured
    observation = replace(observation, state=state)
    model = _action(contract, (1.0, 0.0))
    model = replace(
        model,
        arm_joint_targets={
            "left": np.linspace(1.1, 1.6, 6, dtype=np.float32),
        },
    )
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    resolved, receipt = resolver.resolve(model, observation)

    assert receipt.resolved_combination == "10"
    np.testing.assert_allclose(resolved.arm_joint_targets["left"], measured)


def test_joint_target_entry_requires_finite_measured_qpos_and_is_atomic() -> None:
    contract = _contract()
    observation = _observation()
    state = observation.state.copy()
    state[STATE_ARM_QPOS_SLICES["left"]] = np.nan
    observation = replace(observation, state=state)
    model = replace(
        _action(contract, (1.0, 0.0)),
        arm_joint_targets={"left": np.ones(6, dtype=np.float32)},
    )
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))

    with pytest.raises(ValueError, match="joint hold reference"):
        resolver.resolve(model, observation)
    assert resolver.resolved_combination == "00"


def test_reset_releases_latched_reference_and_debounce_state() -> None:
    contract = _contract(enter_steps=2)
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    observation = _observation()
    resolver.resolve(_action(contract, (1.0, 0.0)), observation)
    resolver.reset()

    _, receipt = resolver.resolve(_action(contract, (1.0, 0.0)), observation)

    assert receipt.resolved_combination == "00"
    assert receipt.side_resolutions[0].consecutive_count == 1


def test_resolution_receipt_hash_authenticates_probability_and_reference_step() -> None:
    contract = _contract()
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    _, receipt = resolver.resolve(
        _action(contract, (1.0, 0.0)),
        _observation(step_index=17),
    )
    payload = receipt.to_dict(include_hash=True)

    assert payload["semantic_content_hash"] == receipt.semantic_content_hash
    assert payload["reference_observation_step_index"] == 17
    assert payload["side_resolutions"][0]["probability"] == 1.0


def test_staged_resolution_is_inert_until_commit_and_can_be_discarded() -> None:
    contract = _contract()
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    observation = _observation()

    discarded = resolver.stage(
        _action(contract, (1.0, 0.0)),
        observation,
    )
    assert resolver.resolved_combination == "00"
    with pytest.raises(RuntimeError, match="already staged"):
        resolver.stage(_action(contract, (1.0, 0.0)), observation)
    resolver.discard(discarded)
    assert resolver.resolved_combination == "00"

    committed = resolver.stage(
        _action(contract, (1.0, 0.0)),
        observation,
    )
    resolver.commit(committed)
    assert resolver.resolved_combination == "10"
    with pytest.raises(RuntimeError, match="not pending"):
        resolver.commit(committed)


def test_reset_invalidates_pending_staged_resolution() -> None:
    contract = _contract()
    resolver = ArmHoldSemanticResolver(contract, active_sides=("left", "right"))
    staged = resolver.stage(
        _action(contract, (1.0, 1.0)),
        _observation(),
    )

    resolver.reset()

    assert resolver.resolved_combination == "00"
    with pytest.raises(RuntimeError, match="not pending"):
        resolver.commit(staged)


def _runtime(
    contract: PolicyArmHoldOutputContract,
    *,
    execution_steps: int | None = None,
    history_binding: ArmHoldResolvedHistoryBinding | None = None,
) -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            arm_hold_output=contract,
            action_chunk_execution_steps=execution_steps,
        ),
        monotonic_clock=_Clock(),
        arm_hold_history_binding=history_binding,
    )


def _history_binding(
    contract: PolicyArmHoldOutputContract,
    *,
    epoch: int = 4,
) -> ArmHoldResolvedHistoryBinding:
    return ArmHoldResolvedHistoryBinding(
        binding_id="policy-source:arm-hold-test",
        epoch=epoch,
        contract_hash=contract.contract_hash,
    )


def test_runtime_resolved_history_is_default_off_and_validated_at_construction() -> None:
    contract = _contract()
    assert _runtime(contract).arm_hold_resolved_history_snapshot() is None

    binding = _history_binding(contract)
    with pytest.raises(ValueError, match="requires arm_hold_output"):
        PolicyDeployRuntime(
            PolicyDeployRuntimeConfig(active_sides=("left", "right")),
            arm_hold_history_binding=binding,
        )
    with pytest.raises(ValueError, match="does not match"):
        _runtime(
            contract,
            history_binding=ArmHoldResolvedHistoryBinding(
                binding_id="policy-source:wrong-contract",
                epoch=0,
                contract_hash="wrong-contract",
            ),
        )


def test_sync_runtime_advances_history_at_each_resolver_commit() -> None:
    contract = _contract()
    binding = _history_binding(contract)
    observation = _observation()
    runtime = _runtime(contract, history_binding=binding)
    bootstrap = runtime.arm_hold_resolved_history_snapshot()
    assert bootstrap is not None
    assert bootstrap.context.available_mask.tolist() == [False, False]
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 0.0), (1.0, 0.0)))),
        observation,
        now=10.0,
    )

    first = runtime.prepare_next_action(observation, preparer=_pipeline())
    after_first = runtime.arm_hold_resolved_history_snapshot()
    second = runtime.prepare_next_action(observation, preparer=_pipeline())
    after_second = runtime.arm_hold_resolved_history_snapshot()

    assert first.arm_hold_resolution is not None
    assert second.arm_hold_resolution is not None
    assert after_first is not None
    assert after_first.last_resolution_sequence == 1
    assert after_first.context.previous_mode.tolist() == [1.0, 0.0]
    assert after_first.context.dwell_steps.tolist() == [1, 1]
    assert after_second is not None
    assert after_second.last_resolution_sequence == 2
    assert after_second.context.previous_mode.tolist() == [1.0, 0.0]
    assert after_second.context.dwell_steps.tolist() == [2, 2]


def test_history_sidecar_does_not_change_sync_action_or_emission_identity() -> None:
    contract = _contract()
    observation = _observation()
    output = _policy_output(contract, ((1.0, 0.0),))
    default_runtime = _runtime(contract)
    history_runtime = _runtime(
        contract,
        history_binding=_history_binding(contract),
    )
    for runtime in (default_runtime, history_runtime):
        prediction = runtime.predict_and_queue(
            _Policy(output),
            observation,
            now=10.0,
        )
        assert prediction.accepted is True

    default_step = default_runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
    )
    history_step = history_runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
    )

    assert default_step.raw_action is not None
    assert history_step.raw_action is not None
    np.testing.assert_array_equal(
        default_step.raw_action.vector,
        history_step.raw_action.vector,
    )
    assert default_step.emission is not None
    assert history_step.emission is not None
    assert (
        default_step.emission.semantic_content_hash
        == history_step.emission.semantic_content_hash
    )
    assert default_runtime.action_queue_pop_count() == 1
    assert history_runtime.action_queue_pop_count() == 1


def test_runtime_semantic_reset_rotates_history_epoch_and_bootstrap() -> None:
    contract = _contract()
    binding = _history_binding(contract, epoch=8)
    observation = _observation()
    runtime = _runtime(contract, history_binding=binding)
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 0.0),))),
        observation,
        now=10.0,
    )
    runtime.prepare_next_action(observation, preparer=_pipeline())

    runtime.reset_arm_hold_semantics()
    reset = runtime.arm_hold_resolved_history_snapshot()

    assert reset is not None
    assert reset.binding.binding_id == binding.binding_id
    assert reset.binding.epoch == 9
    assert reset.context.available_mask.tolist() == [False, False]
    assert reset.context.dwell_steps.tolist() == [0, 0]
    assert reset.last_resolution_sequence is None

    runtime.clear_action_queue()
    runtime.reset_prediction_schedule()
    repeated = runtime.arm_hold_resolved_history_snapshot()

    assert repeated is not None
    assert repeated.binding.semantic_content_hash == (
        reset.binding.semantic_content_hash
    )
    assert repeated.binding.epoch == 9
    assert repeated.context.available_mask.tolist() == [False, False]

    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((0.0, 1.0),))),
        observation,
        now=11.0,
    )
    runtime.prepare_next_action(observation, preparer=_pipeline())
    recommitted = runtime.arm_hold_resolved_history_snapshot()
    assert recommitted is not None
    assert recommitted.binding.epoch == 9
    assert recommitted.last_resolution_sequence == 1

    runtime.reset_arm_hold_semantics()
    second_boundary = runtime.arm_hold_resolved_history_snapshot()
    assert second_boundary is not None
    assert second_boundary.binding.epoch == 10
    assert second_boundary.last_resolution_sequence is None


def test_runtime_semantic_reset_is_idempotent_before_first_history_commit() -> None:
    contract = _contract()
    binding = _history_binding(contract, epoch=12)
    runtime = _runtime(contract, history_binding=binding)

    runtime.reset_arm_hold_semantics()
    runtime.clear_action_queue()
    runtime.reset_prediction_schedule()
    reset = runtime.arm_hold_resolved_history_snapshot()

    assert reset is not None
    assert reset.binding.semantic_content_hash == binding.semantic_content_hash
    assert reset.binding.epoch == 12
    assert reset.context.available_mask.tolist() == [False, False]
    assert reset.last_resolution_sequence is None


def _policy_output(
    contract: PolicyArmHoldOutputContract,
    probabilities: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    model = _action(contract, (0.0, 0.0))
    return {
        "actions": np.stack([model.vector for _ in probabilities]),
        "action_semantics": {
            contract.output_key: np.asarray(probabilities, dtype=np.float32),
            "contract_hash": contract.contract_hash,
        },
    }


def test_runtime_queue_replace_preserves_latch_but_explicit_clear_releases_it() -> None:
    contract = _contract()
    observation = _observation()
    runtime = _runtime(contract)
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 0.0),))),
        observation,
        now=10.0,
    )
    entered = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
    )
    assert entered.arm_hold_resolution is not None
    assert entered.arm_hold_resolution.resolved_combination == "10"

    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((0.5, 0.0),))),
        observation,
        now=11.0,
        queue_policy="replace",
    )
    assert runtime._arm_hold_resolver.resolved_combination == "10"
    held = runtime.prepare_next_action(observation, preparer=_pipeline())
    assert held.arm_hold_resolution is not None
    assert held.arm_hold_resolution.resolved_combination == "10"

    runtime.clear_action_queue()
    assert runtime._arm_hold_resolver.resolved_combination == "00"
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((0.5, 0.0),))),
        observation,
        now=12.0,
    )
    moving = runtime.prepare_next_action(observation, preparer=_pipeline())
    assert moving.arm_hold_resolution is not None
    assert moving.arm_hold_resolution.resolved_combination == "00"


def test_failed_prediction_clears_queue_and_releases_latched_semantics() -> None:
    contract = _contract()
    observation = _observation()
    runtime = _runtime(contract)
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 0.0),))),
        observation,
        now=10.0,
    )
    runtime.prepare_next_action(observation, preparer=_pipeline())
    assert runtime._arm_hold_resolver.resolved_combination == "10"

    failure = runtime.predict_and_queue(
        _Policy({"actions": _action(contract, (0.0, 0.0)).vector}),
        observation,
        now=11.0,
    )

    assert failure.error is not None
    assert runtime.action_queue_length() == 0
    assert runtime._arm_hold_resolver.resolved_combination == "00"


def test_runtime_execution_prefix_keeps_matching_first_semantic_row() -> None:
    contract = _contract()
    observation = _observation()
    runtime = _runtime(contract, execution_steps=1)
    prediction = runtime.predict_and_queue(
        _Policy(
            _policy_output(
                contract,
                ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            )
        ),
        observation,
        now=10.0,
    )

    assert prediction.source_chunk_size == 3
    assert prediction.executed_chunk_size == 1
    assert prediction.actions[0].arm_hold_prediction is not None
    assert prediction.actions[0].arm_hold_prediction.probabilities == (1.0, 0.0)
    emitted = runtime.prepare_next_action(observation, preparer=_pipeline())
    assert emitted.arm_hold_resolution is not None
    assert emitted.arm_hold_resolution.resolved_combination == "10"
    assert runtime.action_queue_length() == 0


def test_runtime_pop_reference_failure_is_no_action_and_does_not_partially_commit() -> None:
    contract = _contract()
    observation = _observation()
    runtime = _runtime(contract, history_binding=_history_binding(contract))
    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 0.0),))),
        observation,
        now=10.0,
    )
    runtime.prepare_next_action(observation, preparer=_pipeline())
    assert runtime._arm_hold_resolver.resolved_combination == "10"

    runtime.predict_and_queue(
        _Policy(_policy_output(contract, ((1.0, 1.0),))),
        observation,
        now=11.0,
    )
    state = observation.state.copy()
    last_action = observation.last_action.copy()
    state[STATE_ARM_POSE_SLICES["right"]] = np.nan
    last_action[ACTION_ARM_POSE_SLICES["right"]] = np.nan
    missing_right = replace(
        observation,
        state=state,
        last_action=last_action,
    )

    failed = runtime.prepare_next_action(
        missing_right,
        preparer=_pipeline(),
    )

    assert isinstance(failed.emission_error, ValueError)
    assert failed.raw_action is None
    assert failed.prepared_action is None
    assert failed.emission is None
    assert failed.model_raw_action is not None
    assert runtime._arm_hold_resolver.resolved_combination == "10"
    committed_history = runtime.arm_hold_resolved_history_snapshot()
    assert committed_history is not None
    assert committed_history.last_resolution_sequence == 1
    assert runtime.action_queue_length() == 0


def test_timed_or_rtc_preparation_fails_closed_instead_of_ignoring_semantics() -> None:
    contract = _contract()
    observation = _observation()
    runtime = _runtime(contract)
    model = _action(contract, (1.0, 0.0))

    result = runtime.prepare_emission_result(
        PolicyActionEmissionResult(raw_action=model),
        observation,
        preparer=_pipeline(),
    )

    assert isinstance(result.emission_error, RuntimeError)
    assert "timed/RTC" in str(result.emission_error)
    assert result.prepared_action is None
    assert runtime._arm_hold_resolver.resolved_combination == "00"


def test_openpi_sim_adapter_rejects_semantics_until_emission_resolution_exists() -> None:
    contract = _contract()
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        arm_hold_output=contract,
    )
    adapter = PolicyAdapterConfig(
        adapter_id="openpi.arm_hold.unsupported",
        model_action_dim=artifact.policy_io_layout.action_dim,
    )

    with pytest.raises(ValueError, match="does not yet own action-emission"):
        OpenPISimAdapterConfig(
            adapter=adapter,
            policy_artifact=artifact,
        )


def _remote_policy(
    *,
    contract: PolicyArmHoldOutputContract | None,
    response: dict[str, object],
) -> RemoteModelPolicy:
    layout = PolicyIOLayout.legacy_v21_q6()
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        arm_hold_output=contract,
    )
    deploy_contract = deploy_contract_from_policy_artifact(artifact)
    metadata: dict[str, object] = {
        "deploy_schema_version": artifact.deploy_schema_version,
        "local_robot": {
            "input_state_key": artifact.state_key,
            "input_state_discrete_key": artifact.state_discrete_key,
            "input_state_dim": layout.state_dim,
            "input_state_discrete_dim": layout.state_discrete_dim,
            "output_action_dim": layout.action_dim,
            "output_action_discrete_dim": layout.action_discrete_dim,
            "active_sides": ["left", "right"],
        },
        "dataset_adapter": {"policy_io_layout_hash": layout.layout_hash},
    }
    if contract is not None:
        metadata["action_semantics"] = {
            "arm_hold_output_key": contract.output_key,
            "arm_hold_output_contract_hash": contract.contract_hash,
        }
    return RemoteModelPolicy(
        transport=_Transport(metadata=metadata, response=response),
        observation_adapter=lambda _value: {},
        config=RemotePolicyConfig(
            action_key=artifact.action_key,
            action_dim=layout.action_dim,
            action_discrete_dim=0,
            contract=deploy_contract,
            arm_hold_output=contract,
        ),
    )


@pytest.mark.parametrize(
    ("response_builder", "error"),
    (
        (
            lambda contract: {
                "actions": np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32),
                "arm_hold_output_contract_hash": contract.contract_hash,
            },
            "missing declared arm-hold key",
        ),
        (
            lambda contract: {
                "actions": np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32),
                contract.output_key: np.zeros((1, 2), dtype=np.float32),
                "arm_hold_output_contract_hash": "wrong",
            },
            "contract hash mismatch",
        ),
        (
            lambda contract: {
                "actions": np.zeros((2, DEPLOY_ACTION_DIM), dtype=np.float32),
                contract.output_key: np.zeros((1, 2), dtype=np.float32),
                "arm_hold_output_contract_hash": contract.contract_hash,
            },
            "shape mismatch",
        ),
        (
            lambda contract: {
                "actions": np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32),
                contract.output_key: np.asarray(((np.nan, 0.0),)),
                "arm_hold_output_contract_hash": contract.contract_hash,
            },
            r"\[0, 1\]",
        ),
        (
            lambda contract: {
                "actions": np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32),
                contract.output_key: np.asarray(((1.1, 0.0),)),
                "arm_hold_output_contract_hash": contract.contract_hash,
            },
            r"\[0, 1\]",
        ),
    ),
)
def test_remote_response_semantics_fail_closed(
    response_builder,
    error: str,
) -> None:
    contract = _contract()
    policy = _remote_policy(
        contract=contract,
        response=response_builder(contract),
    )

    with pytest.raises(ValueError, match=error):
        policy.predict(object())


def test_legacy_remote_client_ignores_extra_semantic_fields_additively() -> None:
    contract = _contract()
    actions = np.zeros((1, DEPLOY_ACTION_DIM), dtype=np.float32)
    policy = _remote_policy(
        contract=None,
        response={
            "actions": actions,
            contract.output_key: np.asarray(((1.0, 1.0),)),
            "arm_hold_output_contract_hash": contract.contract_hash,
        },
    )

    chunk = policy.predict(object())

    np.testing.assert_array_equal(chunk.actions, actions)
    assert chunk.arm_hold_probabilities is None
    assert chunk.arm_hold_contract_hash is None
