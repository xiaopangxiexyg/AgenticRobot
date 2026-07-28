from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from control_station.composition import (
    MujocoHumanOnlyComposition,
    build_cs2a1_mujoco_human_manifest,
    build_mujoco_human_proposal_adapter,
)
from control_station.sources import (
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import MujocoDeployConfig, MujocoDeployRuntime


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


def _timing() -> TimingSpec:
    return TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
    )


def _runtime(
    *,
    clock: ManualClock,
    replay: bool = False,
    composer: bool = False,
) -> tuple[MujocoDeployRuntime, FakeScene]:
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=_timing(),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    return (
        MujocoDeployRuntime(
            rig,
            MujocoDeployConfig(
                runtime_session_id="cs2a1-runtime",
                control_domain_id="cs2a1-mujoco-domain",
                embodiment_instance_id="cs2a1-mujoco-rm65",
                producer_id="synthetic-human-main",
                enable_replay_authority=replay,
                enable_composer_authority=composer,
                decision_hz=30.0,
                authority_policy_id="mujoco-single-human-authority-v1",
                initial_control_mode="human",
            ),
            clock=clock,
        ),
        scene,
    )


def _composition(clock: ManualClock):
    runtime, scene = _runtime(clock=clock)
    source_spec = SyntheticHumanQposSourceSpec()
    adapter = build_mujoco_human_proposal_adapter(runtime)
    manifest = build_cs2a1_mujoco_human_manifest(runtime, adapter, source_spec)
    return (
        MujocoHumanOnlyComposition(
            manifest=manifest,
            runtime=runtime,
            adapter=adapter,
            source_spec=source_spec,
        ),
        SyntheticHumanQposSource(source_spec),
        scene,
    )


def test_cs2a1_exact_binding_is_authoritative_data_plane_not_station_authority() -> None:
    clock = ManualClock()
    composition, _source, _scene = _composition(clock)

    receipt = composition.binding_receipt
    assert receipt.authoritative_data_plane_bound
    assert not receipt.station_authority_lifecycle_bound
    assert composition.manifest.execution_binding.target_kind == "sim"
    assert tuple(slot.source_kind for slot in composition.manifest.source_slots) == (
        "human",
    )
    assert composition.manifest.source_slots[0].supported_uses == ("authority",)
    assert composition.runtime.core.config.initial_control_mode == "human"
    assert composition.runtime.core.config.authority_policy.policy_id == (
        "mujoco-single-human-authority-v1"
    )
    assert "request_motion_authority" not in composition.manifest.allowed_intent_kinds
    assert "station_motion_authority_lifecycle" in composition.manifest.explicit_exclusions
    composition.close()


def test_cs2a1_synthetic_human_uses_existing_authoritative_mujoco_core() -> None:
    clock = ManualClock()
    composition, source, scene = _composition(clock)
    composition.start()
    emission = source.emit(
        {
            "left_arm": np.full(6, 0.001, dtype=float),
            "right_arm": np.full(6, -0.001, dtype=float),
        },
        emitted_at=clock(),
    )
    submission = composition.submit_emission(
        emission,
        effective_from=clock(),
        effective_until=clock() + 0.1,
        target_effective_at=clock(),
    )
    assert submission.accepted
    assert submission.adapter_result.proposal_draft is not None
    assert submission.adapter_result.proposal_draft.producer_kind == "human"
    assert source.next_sequence == 1

    step = composition.step(now=clock(), physics_ticks_after_dispatch=1)
    assert step.status == "dispatched"
    assert scene.physics_commands
    assert step.decision is not None
    assert np.allclose(
        step.decision.selected_action.value("left_arm").values,
        0.001,
    )
    assert np.allclose(
        step.decision.selected_action.value("right_arm").values,
        -0.001,
    )

    facts = composition.runtime.trace_facts
    event_types = tuple(fact.event_type for fact in facts)
    assert event_types.count("human_command_emission") == 1
    assert event_types.count("source_adapter_receipt") == 1
    assert event_types.count("source_validation") == 1
    assert event_types.count("action_proposal") == 1
    assert event_types.count("arbitration_decision") == 1
    assert "policy_output" not in event_types
    proposal_fact = next(fact for fact in facts if fact.event_type == "action_proposal")
    assert proposal_fact.payload["producer_kind"] == "human"
    decision_fact = next(
        fact for fact in facts if fact.event_type == "arbitration_decision"
    )
    assert submission.proposal_admission is not None
    assert decision_fact.payload["selected_by_group"]["arms"] == (
        submission.proposal_admission.proposal_id
    )
    composition.close()


def test_cs2a1_rejects_manifest_drift_and_additional_authority_sources() -> None:
    clock = ManualClock()
    runtime, _scene = _runtime(clock=clock)
    source_spec = SyntheticHumanQposSourceSpec()
    adapter = build_mujoco_human_proposal_adapter(runtime)
    manifest = build_cs2a1_mujoco_human_manifest(runtime, adapter, source_spec)
    mismatched = replace(manifest, capture_policy_id="capture:other")
    with pytest.raises(ValueError, match="does not exactly bind"):
        MujocoHumanOnlyComposition(
            manifest=mismatched,
            runtime=runtime,
            adapter=adapter,
            source_spec=source_spec,
        )
    runtime.close()

    replay_runtime, _scene = _runtime(clock=clock, replay=True)
    replay_adapter = build_mujoco_human_proposal_adapter(replay_runtime)
    replay_manifest = build_cs2a1_mujoco_human_manifest(
        replay_runtime,
        replay_adapter,
        source_spec,
    )
    with pytest.raises(ValueError, match="replay authority"):
        MujocoHumanOnlyComposition(
            manifest=replay_manifest,
            runtime=replay_runtime,
            adapter=replay_adapter,
            source_spec=source_spec,
        )
    replay_runtime.close()


def test_synthetic_human_source_fails_closed_without_consuming_sequence() -> None:
    source = SyntheticHumanQposSource(SyntheticHumanQposSourceSpec())
    with pytest.raises(ValueError, match="exactly match"):
        source.emit(
            {"left_arm": np.zeros(6, dtype=float)},
            emitted_at=10.0,
        )
    assert source.next_sequence == 0
    with pytest.raises(ValueError, match="finite qpos"):
        source.emit(
            {
                "left_arm": np.full(6, np.nan),
                "right_arm": np.zeros(6, dtype=float),
            },
            emitted_at=10.0,
        )
    assert source.next_sequence == 0


def test_cs2a1_retains_global_arbiter_ceiling() -> None:
    with pytest.raises(ValueError, match=r"\(0, 60\]"):
        MujocoDeployConfig(
            producer_id="synthetic-human-main",
            decision_hz=60.1,
        )
