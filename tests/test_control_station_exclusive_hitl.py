from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import control_station.sim_hil as sim_hil_module
from control_station.sim_hil import MujocoSimHILApplicationConfig
from control_station.sim_hil_acceptance import (
    SimHILAcceptanceController,
    SimHILAcceptanceIntent,
    SimHILAcceptanceTraceCapture,
)
from control_station.sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
)
from tests.test_control_station_sim_hil_acceptance import _FakeViveSetup
from tests.test_control_station_sim_hil_allocation_profile import (
    _VirtualAllocationPolicySource,
)
from tests.test_control_station_sim_hil_device_adapters import (
    _install_local_virtual_device_io,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.slow_sim]


class _ResetCountingVirtualPolicy(_VirtualAllocationPolicySource):
    instances: list["_ResetCountingVirtualPolicy"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_count = 0
        self.__class__.instances.append(self)

    def reset_for_activation(self) -> None:
        self.reset_count += 1
        super().reset_for_activation()


def _intent(
    controller: SimHILAcceptanceController,
    *,
    sequence: int,
    kind: str,
    control_mode: str | None = None,
) -> SimHILAcceptanceIntent:
    return SimHILAcceptanceIntent(
        client_id="exclusive-hitl-virtual-test",
        client_sequence=sequence,
        expected_revision=controller.status()["revision"],
        intent_kind=kind,
        control_mode=control_mode,
        reason="virtual_test",
    )


def _execute(
    controller: SimHILAcceptanceController,
    intent: SimHILAcceptanceIntent,
) -> dict[str, object]:
    receipt = controller.submit(intent)
    assert receipt.accepted, receipt
    result = controller.process_next()
    assert result is not None
    return result


def test_exclusive_hitl_virtual_policy_and_devices_switch_through_mujoco(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise trigger -> HOLD/Home -> 0/1 compose -> fresh source -> target."""

    pytest.importorskip("mujoco")
    coordinate_profile = Path(
        "teleop_records/control_station_vive_coordinate_profiles/"
        "vive_rm65_virtual_desk_coordinate_v1.json"
    ).resolve()
    assert coordinate_profile.is_file()
    concrete_human_builder = (
        sim_hil_module._build_passive_single_vive_gripper_component_source
    )
    human_sources: list[Any] = []

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
    monkeypatch.setattr(
        "sim.mujoco_temp_pipeline.hil_policy.TempPPOHILPolicySource",
        _ResetCountingVirtualPolicy,
    )
    _ResetCountingVirtualPolicy.instances.clear()
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="exclusive-hitl-virtual-mujoco",
        max_spool_bytes=64 << 20,
    )
    setup = _FakeViveSetup(
        input_ready=True,
        current_profile_path=coordinate_profile,
    )

    def application_factory(path: Path, role: str, alpha: float):
        assert Path(path) == coordinate_profile
        return sim_hil_module._build_application(
            MujocoSimHILApplicationConfig(
                enabled=True,
                control_profile_id=(
                    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL
                ),
                initial_human_alpha=float(alpha),
                defer_human_input=True,
                vive_coordinate_profile_path=coordinate_profile,
                vive_source_tracker_role=role,
                gripper_input_source="angle_trigger",
                device="cpu",
                realtime_worker=False,
                mujoco_viewer_enabled=False,
                # Keep activation execution on the production-default budget.
                # A cold isolated Profile can exceed one second while still
                # producing the same admitted target; human emission remains
                # independently fail-closed by the one-second timeout below.
                warmup_timeout_sec=5.0,
                human_emission_timeout_sec=1.0,
                compose_target_hz=15.0,
            ),
            trace_sink=capture.admit_trace,
        )

    controller = SimHILAcceptanceController(
        None,
        trace_capture=capture,
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

    def step_until_proposal_admitted():
        last_cycle = None
        for _ in range(10):
            assert app is not None
            due = app.next_decision_at
            if due is not None:
                time.sleep(max(0.0, due - app.clock()))
            last_cycle = app.step()
            status = last_cycle.composition_step.status
            if status == "proposal_admitted":
                return last_cycle
            assert status == "no_action"
        pytest.fail(
            "asynchronous allocation context produced no admitted proposal; "
            f"last_cycle={last_cycle!r}"
        )

    try:
        started = _execute(
            controller,
            _intent(
                controller,
                sequence=1,
                kind="start_runtime",
                control_mode="ppo_only",
            ),
        )
        assert started["status"] == "running"
        app = controller.current_application
        assert app is not None
        assert app.human_input_attached is False
        assert app.status()["composition"]["mix"]["mode"] == "policy_only"
        assert app.status()["composition"]["allocation_operations"][
            "allocation_override_mode"
        ] == "base_only"
        assert set(
            app.status()["composition"]["allocation_operations"][
                "supported_signals"
            ]
        ) == {
            "allocation_base_only",
            "allocation_absolute_overlay_owner",
        }

        due = app.next_decision_at
        if due is not None:
            time.sleep(max(0.0, due - app.clock()))
        policy_cycle = app.step()
        assert policy_cycle.execution.status in {
            "dispatched",
            "execution_kept_existing",
            "not_due",
        }

        takeover = _execute(
            controller,
            _intent(
                controller,
                sequence=2,
                kind="request_human_takeover",
            ),
        )
        assert takeover["status"] == "switch_ready", json.dumps(
            {"takeover": takeover, "application": app.status()},
            indent=2,
            default=str,
        )
        assert takeover["exclusive_hitl_switch"]["target_source"] == "human"
        assert takeover["exclusive_hitl_switch"]["home_capture"] is not None
        human_evidence_refs = set(
            takeover["exclusive_hitl_switch"]["evidence_refs"]
        )
        assert any(value.startswith("component-target-admission:") for value in human_evidence_refs)
        assert any(value.startswith("command:") for value in human_evidence_refs)
        assert any(value.startswith("dispatch-receipt:") for value in human_evidence_refs)
        assert app.human_input_attached is True
        assert setup.closed is True
        assert len(human_sources) == 1
        assert app.status()["composition"]["allocation_operations"][
            "allocation_override_mode"
        ] == "absolute_overlay_owner"
        right_retargeter = next(
            item
            for item in human_sources[0].session.retargeters
            if getattr(getattr(item, "config", None), "side", None) == "right"
            and hasattr(item, "last_target_pose_desk")
        )
        np.testing.assert_allclose(
            right_retargeter.last_target_pose_desk,
            np.asarray(
                takeover["exclusive_hitl_switch"]["home_capture"]
                ["robot_home_pose_desk_by_side"]["right"],
                dtype=float,
            ),
            rtol=0.0,
            atol=1e-8,
        )

        human_cycle = step_until_proposal_admitted()
        human_publication = human_cycle.composition_step
        assert human_publication.status == "proposal_admitted"
        assert set(human_publication.selected_mode_by_component.values()) == {
            "absolute_overlay_owner"
        }
        human_submission = human_publication.submission
        assert human_submission is not None
        assert (
            human_submission.composition_evaluation.composed_candidate
            is not None
        )
        assert "human_absolute" in (
            human_submission.input_snapshot.selected_input_by_role
        )

        reset_count_before_resume = (
            _ResetCountingVirtualPolicy.instances[0].reset_count
        )
        resume = _execute(
            controller,
            _intent(
                controller,
                sequence=3,
                kind="request_policy_resume",
            ),
        )
        assert resume["status"] == "switch_ready"
        switch = resume["exclusive_hitl_switch"]
        assert switch["target_source"] == "policy"
        assert switch["source_reset_performed"] is True
        policy_evidence_refs = set(switch["evidence_refs"])
        assert any(value.startswith("component-target-admission:") for value in policy_evidence_refs)
        assert any(value.startswith("command:") for value in policy_evidence_refs)
        assert any(value.startswith("dispatch-receipt:") for value in policy_evidence_refs)
        assert (
            _ResetCountingVirtualPolicy.instances[0].reset_count
            > reset_count_before_resume
        )
        assert app.status()["composition"]["allocation_operations"][
            "allocation_override_mode"
        ] == "base_only"

        resumed_cycle = step_until_proposal_admitted()
        resumed_publication = resumed_cycle.composition_step
        assert resumed_publication.status == "proposal_admitted"
        assert set(resumed_publication.selected_mode_by_component.values()) == {
            "base_only"
        }
        resumed_submission = resumed_publication.submission
        assert resumed_submission is not None
        selected = resumed_submission.input_snapshot.selected_input_by_role
        assert "policy" in selected
        composed = resumed_submission.composition_evaluation.composed_candidate
        assert composed is not None
        for component_id, value in composed.values_by_component.items():
            np.testing.assert_allclose(
                value.values,
                selected["policy"].action.values_by_component[component_id].values,
                rtol=0.0,
                atol=1e-10,
            )

        switch_facts = [
            record.fact
            for record in capture.trace_runtime.records()
            if record.fact.event_type == "exclusive_hitl_switch"
        ]
        assert [fact.payload["target_source"] for fact in switch_facts] == [
            "human",
            "policy",
        ]
        assert switch_facts[0].payload["home_capture"][
            "human_home_pose_desk_by_side"
        ]
        assert switch_facts[1].payload["source_reset_performed"] is True
    finally:
        if app is not None:
            app.close()
        capture.abort(now=time.monotonic())
        capture.close(now=time.monotonic())
