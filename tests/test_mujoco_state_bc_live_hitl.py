from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import control_station.sim_hil as sim_hil_module
from control_station.sim_hil import MujocoSimHILApplicationConfig
from control_station.sim_hil_acceptance import (
    SimHILAcceptanceController,
    SimHILAcceptanceIntent,
)
from control_station.sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
)
from experiments.sim_learning.state_bc.contracts import (
    StateBCArtifactMetadata,
    StateBCNormalization,
)
from experiments.sim_learning.state_bc.model import StateBCModel
from tests.test_control_station_sim_hil_acceptance import _FakeViveSetup
from tests.test_control_station_sim_hil_device_adapters import (
    _install_local_virtual_device_io,
)


pytestmark = [pytest.mark.slow_sim]


def _identity_artifact(tmp_path: Path) -> Path:
    model = StateBCModel(
        weights=np.eye(16, dtype=np.float32),
        bias=np.zeros(16, dtype=np.float32),
        normalization=StateBCNormalization(
            mean=np.zeros(16, dtype=np.float32),
            scale=np.ones(16, dtype=np.float32),
        ),
        artifact_metadata=StateBCArtifactMetadata(
            selector_id="mujoco-live-hitl-fixture",
            train_episode_count=1,
            test_episode_count=0,
            train_frame_count=1,
            test_frame_count=0,
        ),
    )
    return model.save(tmp_path / "state_bc_identity.npz")


def _execute(
    controller: SimHILAcceptanceController,
    *,
    sequence: int,
    intent_kind: str,
    control_mode: str | None = None,
) -> dict[str, object]:
    intent = SimHILAcceptanceIntent(
        client_id="state-bc-live-hitl-test",
        client_sequence=sequence,
        expected_revision=controller.status()["revision"],
        intent_kind=intent_kind,
        control_mode=control_mode,
        reason="state_bc_live_hitl_test",
    )
    receipt = controller.submit(intent)
    assert receipt.accepted, receipt
    result = controller.process_next()
    assert result is not None
    assert result.get("status") not in {"failed", "faulted"}, result
    return result


def test_state_bc_policy_reaches_mujoco_and_supports_human_handback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise Policy -> Human -> Policy through one live Sim-HIL graph."""

    pytest.importorskip("mujoco")
    artifact = _identity_artifact(tmp_path)
    coordinate_profile = Path(
        "teleop_records/control_station_vive_coordinate_profiles/"
        "vive_rm65_virtual_desk_coordinate_v1.json"
    ).resolve()
    assert coordinate_profile.is_file()
    concrete_human_builder = (
        sim_hil_module._build_passive_single_vive_gripper_component_source
    )
    human_sources: list[object] = []

    def local_human_builder(**kwargs):
        source, source_spec = concrete_human_builder(**kwargs)
        _install_local_virtual_device_io(
            monkeypatch,
            source,
            input_kind="angle_trigger",
        )
        human_sources.append(source)
        return source, source_spec

    monkeypatch.setattr(
        sim_hil_module,
        "_build_passive_single_vive_gripper_component_source",
        local_human_builder,
    )
    setup = _FakeViveSetup(
        input_ready=True,
        current_profile_path=coordinate_profile,
    )

    def application_factory(_path: Path, _role: str, _alpha: float):
        return sim_hil_module._build_application(
            MujocoSimHILApplicationConfig(
                enabled=True,
                policy_backend="state_bc",
                state_bc_artifact_path=artifact,
                control_profile_id=(
                    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
                ),
                initial_human_alpha=0.0,
                defer_human_input=True,
                vive_coordinate_profile_path=coordinate_profile,
                gripper_input_source="angle_trigger",
                device="cpu",
                realtime_worker=False,
                mujoco_viewer_enabled=False,
                warmup_timeout_sec=3.0,
                human_emission_timeout_sec=1.0,
                compose_target_hz=15.0,
            )
        )

    controller = SimHILAcceptanceController(
        None,
        application_factory=application_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
            current_profile_path=coordinate_profile,
        ),
        coordinate_profile_path=coordinate_profile,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )
    app = None
    try:
        started = _execute(
            controller,
            sequence=1,
            intent_kind="start_runtime",
            control_mode="ppo_only",
        )
        assert started["status"] == "running"
        app = controller.current_application
        assert app is not None
        assert app.policy_source.status()["backend"] == "state_bc"

        policy_cycle = app.step()
        assert policy_cycle.execution.status in {
            "dispatched",
            "execution_kept_existing",
            "not_due",
        }

        takeover = _execute(
            controller,
            sequence=2,
            intent_kind="request_human_takeover",
        )
        assert takeover["status"] == "switch_ready", takeover
        assert takeover["exclusive_hitl_switch"]["target_source"] == "human"
        assert app.human_input_attached is True
        assert len(human_sources) == 1
        human_cycle = app.step()
        assert human_cycle.composition_step.status == "proposal_admitted"

        resume = _execute(
            controller,
            sequence=3,
            intent_kind="request_policy_resume",
        )
        assert resume["status"] == "switch_ready", resume
        assert resume["exclusive_hitl_switch"]["target_source"] == "policy"
        assert app.policy_source.status()["backend"] == "state_bc"
        resumed_cycle = app.step()
        assert resumed_cycle.composition_step.status == "proposal_admitted"
        assert app.policy_source.last_emission is not None
    finally:
        if app is not None:
            app.close()
