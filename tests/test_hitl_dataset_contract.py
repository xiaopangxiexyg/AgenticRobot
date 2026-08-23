from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.hitl import (
    EXCLUSIVE_HITL_DATASET_METADATA_KEY,
    EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
    ExclusiveHITLDatasetContract,
    ExclusiveHITLSampleLineage,
)
from dataset_interface.hitl import (
    ExclusiveHITLDatasetReadError,
    read_exclusive_hitl_dataset_contract,
    validate_exclusive_hitl_dataset_append,
    validate_exclusive_hitl_dataset_rows,
)
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilder,
    _position_reference_command,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.dataset_recorder import (
    _exclusive_hitl_dataset_metadata,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand
from teleop_runtime.session import SessionStepResult


def _dataset_contract(**overrides) -> ExclusiveHITLDatasetContract:
    values = {
        "policy_mode": "normal",
        "initial_source": "policy",
        "active_sides": ("right",),
        "policy_source_kind": "openpi_artifact",
        "rtc_no_action": False,
        "takeover_key": "u",
        "policy_resume_key": "i",
        "policy_artifact_contract_hash": "artifact-contract-hash",
        "policy_io_layout_hash": "policy-layout-hash",
        "model_id": "openpi-fixture",
        "checkpoint_step": 123,
        "policy_prompt": "write the requested text",
        "action_hz": 15.0,
        "policy_hz": 10.0,
        "action_chunk_execution_steps": 8,
        "prediction_request_queue_threshold": 4,
    }
    values.update(overrides)
    return ExclusiveHITLDatasetContract(**values)


def _human_lineage(*, switch_id: str | None = None) -> ExclusiveHITLSampleLineage:
    return ExclusiveHITLSampleLineage(
        source="human",
        policy_mode="normal",
        rtc_no_action=False,
        switch_id=switch_id,
    )


def _policy_lineage(*, switch_id: str | None = None) -> ExclusiveHITLSampleLineage:
    return ExclusiveHITLSampleLineage(
        source="policy",
        policy_mode="normal",
        rtc_no_action=False,
        switch_id=switch_id,
        intent_id="intent:1",
        intent_hash="intent-hash:1",
        policy_emission_id="emission:1",
        policy_emission_hash="emission-hash:1",
        policy_output_id="output:1",
        policy_inference_id="inference:1",
        policy_target_effective_at=10.0,
        policy_effective_until=10.2,
        composition_input_id="composition-input:1",
        composition_input_hash="composition-input-hash:1",
        composition_evaluation_id="composition-evaluation:1",
        composition_evaluation_hash="composition-evaluation-hash:1",
        composition_spec_hash="composition-spec-hash:1",
        composed_candidate_hash="candidate-hash:1",
    )


def _row(
    lineage: ExclusiveHITLSampleLineage,
    *,
    with_position_reference: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        EXCLUSIVE_HITL_SAMPLE_METADATA_KEY: lineage.to_payload(),
        "command_metadata": lineage.to_command_metadata(),
    }
    if with_position_reference:
        metadata["position_reference"] = {
            "epoch": 7,
            "sequence": 3,
            "frame_metadata": {
                EXCLUSIVE_HITL_SAMPLE_METADATA_KEY: lineage.to_payload(),
            },
        }
    return {"sample_metadata_json": json.dumps(metadata)}


def test_dataset_contract_round_trips_and_binds_writer_metadata() -> None:
    contract = _dataset_contract()
    payload = contract.to_payload()

    assert ExclusiveHITLDatasetContract.from_payload(payload) == contract
    assert contract.policy_identity_status == "artifact_bound"
    assert payload["selection_semantics"] == "exclusive_single_source_no_blending"
    assert payload["counterfactual_policy_action_recorded"] is False
    assert (
        read_exclusive_hitl_dataset_contract(
            {EXCLUSIVE_HITL_DATASET_METADATA_KEY: payload}
        )
        == contract
    )

    args = SimpleNamespace(
        hitl_policy_control=True,
        _exclusive_hitl_dataset_contract=contract,
    )
    assert _exclusive_hitl_dataset_metadata(args) == {
        EXCLUSIVE_HITL_DATASET_METADATA_KEY: payload
    }


def test_enabled_recorder_fails_closed_without_hitl_dataset_contract() -> None:
    with pytest.raises(RuntimeError, match="missing its dataset contract"):
        _exclusive_hitl_dataset_metadata(SimpleNamespace(hitl_policy_control=True))


def test_append_admission_rejects_populated_dataset_without_hitl_identity(
    tmp_path,
) -> None:
    validate_exclusive_hitl_dataset_append(tmp_path, _dataset_contract())
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps({"frame_count": 1, "episode_count": 1}),
        encoding="utf-8",
    )

    with pytest.raises(ExclusiveHITLDatasetReadError, match="populated dataset"):
        validate_exclusive_hitl_dataset_append(tmp_path, _dataset_contract())


def test_append_admission_rejects_different_bound_hitl_identity(tmp_path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    existing = _dataset_contract(initial_source="human")
    (meta / "info.json").write_text(
        json.dumps(
            {
                "frame_count": 0,
                "episode_count": 0,
                EXCLUSIVE_HITL_DATASET_METADATA_KEY: existing.to_payload(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExclusiveHITLDatasetReadError, match="differs"):
        validate_exclusive_hitl_dataset_append(tmp_path, _dataset_contract())


def test_policy_sample_requires_complete_emission_and_composition_lineage() -> None:
    metadata = _policy_lineage().to_command_metadata()
    metadata.pop("hitl_composition_evaluation_hash")
    metadata.pop(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)

    with pytest.raises(ValueError, match="composition_evaluation_hash"):
        ExclusiveHITLSampleLineage.from_command_metadata(metadata)


def test_reader_validates_human_policy_transition_and_qref_binding() -> None:
    contract = _dataset_contract()
    human = _human_lineage(switch_id="switch:1")
    policy = _policy_lineage(switch_id="switch:2")

    summary = validate_exclusive_hitl_dataset_rows(
        {EXCLUSIVE_HITL_DATASET_METADATA_KEY: contract.to_payload()},
        (
            _row(human, with_position_reference=True),
            _row(policy, with_position_reference=True),
        ),
    )

    assert summary.sample_count == 2
    assert summary.human_sample_count == 1
    assert summary.policy_sample_count == 1
    assert summary.source_transition_count == 1
    assert summary.switch_count == 2
    assert summary.position_reference_sample_count == 2


def test_reader_rejects_position_reference_with_different_causal_source() -> None:
    contract = _dataset_contract()
    human = _human_lineage()
    policy = _policy_lineage()
    row = _row(human, with_position_reference=True)
    metadata = json.loads(str(row["sample_metadata_json"]))
    metadata["position_reference"]["frame_metadata"][
        EXCLUSIVE_HITL_SAMPLE_METADATA_KEY
    ] = policy.to_payload()
    row["sample_metadata_json"] = json.dumps(metadata)

    with pytest.raises(ExclusiveHITLDatasetReadError, match="disagrees"):
        validate_exclusive_hitl_dataset_rows(
            {EXCLUSIVE_HITL_DATASET_METADATA_KEY: contract.to_payload()},
            (row,),
        )


def test_position_reference_command_uses_reference_lineage_not_current_source() -> None:
    current_policy = _policy_lineage()
    reference_human = _human_lineage(switch_id="switch:takeover")
    current = TeleopCommand(
        timestamp=11.0,
        metadata=current_policy.to_command_metadata(),
    )
    current.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.ones(6, dtype=float),
        )
    )
    reference = {
        "epoch": 8,
        "sequence": 4,
        "published_timestamp": 10.5,
        "q_ref_by_side": {"right": np.zeros(6, dtype=float)},
        "pose_ref_by_side": {"right": np.eye(4, dtype=float)},
        "frame_metadata": {
            EXCLUSIVE_HITL_SAMPLE_METADATA_KEY: reference_human.to_payload(),
        },
    }

    restored = _position_reference_command(
        current,
        reference,
        active_sides=("right",),
    )

    assert restored.metadata["hitl_exclusive_source"] == "human"
    assert restored.metadata["hitl_switch_id"] == "switch:takeover"
    assert "policy_emission_id" not in restored.metadata
    assert (
        ExclusiveHITLSampleLineage.from_command_metadata(restored.metadata)
        == reference_human
    )
    np.testing.assert_array_equal(
        restored.commands["right_arm"].value,
        np.zeros(6, dtype=float),
    )


def test_formal_compact_sample_metadata_keeps_canonical_hitl_lineage() -> None:
    lineage = _human_lineage(switch_id="switch:formal")
    command_metadata = lineage.to_command_metadata()
    command_metadata.pop(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)
    command = TeleopCommand(timestamp=20.0, metadata=command_metadata)
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.zeros(6, dtype=float),
        )
    )
    step = SessionStepResult(
        timestamp=20.0,
        state=TeleopRunState.RUNNING,
        control_hz=45.0,
        command=command,
        command_dispatched=True,
    )

    metadata = CollectorSampleBuilder._default_metadata(
        step,
        SimpleNamespace(active_sides=("right",)),
        {"right": {"current_qpos": np.zeros(6, dtype=float)}},
        {"right": {}},
        {},
        include_debug_metadata=False,
    )

    assert metadata[EXCLUSIVE_HITL_SAMPLE_METADATA_KEY] == lineage.to_payload()
    assert metadata["command_metadata"]["hitl_exclusive_source"] == "human"
    assert metadata["command_metadata"]["hitl_policy_mode"] == "normal"
