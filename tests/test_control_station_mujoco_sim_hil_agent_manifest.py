from __future__ import annotations

from types import SimpleNamespace

import pytest

import control_station.composition.mujoco_sim_hil as sim_hil
from control_station.composition.mujoco_sim_hil import (
    MUJOCO_SIM_HIL_AGENT_SKILL_EXECUTION_MANIFEST_ID_V1,
    MUJOCO_SIM_HIL_AGENT_SKILL_STATION_MANIFEST_ID_V1,
    MujocoSimHILComposition,
    build_mujoco_sim_hil_agent_skill_execution_manifest,
    build_mujoco_sim_hil_agent_skill_station_manifest,
)
from robot_motion.action import BIMANUAL_EE_POSE_V1


def _hashed(value: str) -> SimpleNamespace:
    return SimpleNamespace(semantic_content_hash=value)


def _composition(*, human_alpha: float = 0.0) -> MujocoSimHILComposition:
    runtime_config = SimpleNamespace(
        runtime_session_id="sim-hil:agent-skill:test:1",
        control_domain_id="mujoco-sim-hil",
        embodiment_instance_id="mujoco-rm65-sim-hil",
        producer_epoch=1,
        composer_producer_id="control-station-sim-hil-composer",
        enable_composer_authority=True,
        enable_primary_policy_authority=False,
        enable_replay_authority=False,
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        component_execution_binding=_hashed("component-execution-binding:test"),
        catalog=SimpleNamespace(
            action_space=BIMANUAL_EE_POSE_V1,
            semantic_content_hash="catalog:test",
        ),
        action_schema=_hashed("action-schema:test"),
        component_target_capability=_hashed("target-capability:test"),
        target_runtime_action_contract=_hashed("target-action-contract:test"),
        execution_capability=_hashed("execution-capability:test"),
        reference_activation_spec=SimpleNamespace(
            activation_mode="immediate_target"
        ),
        authoritative_port=SimpleNamespace(
            profile=SimpleNamespace(
                selection_group_ids=("left_arm", "right_arm"),
                authority_policy_hash="authority-policy:test",
                decision_schedule_hash="decision-schedule:test",
            ),
        ),
    )
    policy_binding = SimpleNamespace(
        source_slot_id="sim-hil-temp-ppo",
        producer_id="sim-hil-temp-ppo",
        producer_epoch=1,
        semantic_content_hash="policy-binding:test",
    )
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = SimpleNamespace(
        initial_human_alpha=human_alpha,
        human_producer_id="human-passive",
        human_producer_epoch=1,
    )
    composition.policy_input_adapter = SimpleNamespace(binding=policy_binding)
    composition.human_source = SimpleNamespace(
        binding_receipt=SimpleNamespace(
            rig_absent=True,
            authoritative=False,
        )
    )
    composition.human_source_spec = SimpleNamespace(
        source_id="passive-teleop-components",
        semantic_content_hash="human-source-spec:test",
    )
    composition.human_adapter = SimpleNamespace(
        contract_hash="human-adapter:test"
    )
    composition.contracts = SimpleNamespace(
        semantic_content_hash="composition-contracts:test",
        composition_spec=_hashed("composition-spec:test"),
        authority_gate_spec=_hashed("authority-gate:test"),
        strategy_spec=_hashed("strategy-spec:test"),
    )
    return composition


def test_policy_only_station_manifest_has_one_authority_and_optional_human() -> None:
    manifest = build_mujoco_sim_hil_agent_skill_station_manifest(
        _composition()
    )
    assert manifest.manifest_id == (
        MUJOCO_SIM_HIL_AGENT_SKILL_STATION_MANIFEST_ID_V1
    )
    assert len(manifest.source_slots) == 3
    human = next(
        slot for slot in manifest.source_slots if slot.source_kind == "human"
    )
    composer = next(
        slot for slot in manifest.source_slots if slot.source_kind == "composite"
    )
    assert not human.required_for_run
    assert human.supported_uses == ("contributor",)
    assert composer.supported_uses == ("authority", "candidate")
    assert "estop" not in manifest.allowed_intent_kinds
    assert "agent_estop" in manifest.explicit_exclusions
    assert manifest.execution_binding.target_kind == "sim"


def test_station_manifest_rejects_human_mix_or_attached_human_source() -> None:
    with pytest.raises(ValueError, match="policy-only alpha=0"):
        build_mujoco_sim_hil_agent_skill_station_manifest(
            _composition(human_alpha=0.5)
        )
    composition = _composition()
    composition.human_source = SimpleNamespace(
        binding_receipt=SimpleNamespace(
            rig_absent=False,
            authoritative=False,
        )
    )
    with pytest.raises(ValueError, match="unattached"):
        build_mujoco_sim_hil_agent_skill_station_manifest(composition)


def test_execution_manifest_maps_only_existing_composer_authority(
    monkeypatch,
) -> None:
    composition = _composition()
    station = build_mujoco_sim_hil_agent_skill_station_manifest(composition)
    captured = {}

    def capture_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_id=kwargs["manifest_id"],
            semantic_content_hash="execution-manifest:test",
        )

    monkeypatch.setattr(
        sim_hil,
        "build_control_execution_manifest",
        capture_builder,
    )
    manifest = build_mujoco_sim_hil_agent_skill_execution_manifest(
        composition,
        station,
    )
    assert manifest.manifest_id == (
        MUJOCO_SIM_HIL_AGENT_SKILL_EXECUTION_MANIFEST_ID_V1
    )
    assert captured["motion_authority_sources"] == (
        "composer:control-station-sim-hil-composer",
    )
    assert captured["human_motion_proposal_status"] == "not_applicable"
    assert captured["safety_control_requests"] == ("hold",)
    excluded = {item.capability_id for item in captured["exclusions"]}
    assert {
        "attached_human_device",
        "raw_primary_policy_authority",
        "real_backend",
        "replay_authority",
        "second_dispatch_owner",
    } <= excluded
