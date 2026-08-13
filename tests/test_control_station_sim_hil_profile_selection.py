from __future__ import annotations

from types import SimpleNamespace

import pytest

from control_station.sim_hil_profile.config import MujocoSimHILApplicationConfig
from control_station.sim_hil_profile.control_cycle import (
    SIM_HIL_POLICY_SOURCE_ALWAYS,
    FixedAlphaCompatibilityControlCyclePort,
    SimHILControlCyclePortSpec,
)
from control_station.sim_hil_profile.factory import build_application_graph
from control_station.sim_hil_profile.profile_selection import (
    SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
    SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
    SimHILApplicationControlProfile,
    SimHILControlProfileAssembly,
    SimHILControlProfileBinding,
    select_sim_hil_application_control_profile,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _session():
    return SimpleNamespace(
        step=lambda **_kwargs: None,
        preflight=lambda **_kwargs: None,
        start=lambda: None,
        request_hold=lambda **_kwargs: None,
        rearm=lambda **_kwargs: None,
        close=lambda: None,
    )


def _assembly() -> SimHILControlProfileAssembly:
    return SimHILControlProfileAssembly(
        compatibility_session=_session(),
        runtime=SimpleNamespace(name="runtime"),
        policy_source=SimpleNamespace(name="policy"),
        human_source=SimpleNamespace(name="human"),
        readiness_provider=SimpleNamespace(name="readiness"),
    )


class _Port:
    def __init__(self, cycle_kind: str) -> None:
        self.spec = SimHILControlCyclePortSpec(
            port_id=f"test:{cycle_kind}",
            cycle_kind=cycle_kind,
            policy_source_requirement=SIM_HIL_POLICY_SOURCE_ALWAYS,
            application_blend_failure_fallback_enabled=False,
            application_human_timeout_fallback_enabled=False,
        )

    def activate(self, **_kwargs):
        return None

    def deactivate(self, **_kwargs):
        return None

    def run_cycle(self, **_kwargs):
        return None

    def close(self):
        return None


def test_profile_selection_defaults_to_exact_fixed_alpha_compatibility() -> None:
    config = MujocoSimHILApplicationConfig()
    assembly = _assembly()

    profile = select_sim_hil_application_control_profile(
        config.control_profile_id,
        assembly=assembly,
    )

    assert config.control_profile_id == (
        SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY
    )
    assert profile.session is assembly.compatibility_session
    assert isinstance(
        profile.control_cycle_port,
        FixedAlphaCompatibilityControlCyclePort,
    )
    assert profile.control_cycle_port.composition is profile.session


def test_exclusive_hitl_is_not_registered_as_fixed_alpha_compatibility() -> None:
    with pytest.raises(ValueError, match="No Sim-HIL application control binding"):
        select_sim_hil_application_control_profile(
            SIM_HIL_CONTROL_PROFILE_EXCLUSIVE_TRUNCATED_HITL,
            assembly=_assembly(),
        )


def test_profile_selection_constructs_only_the_explicit_binding() -> None:
    calls = []
    selected = SimHILControlProfileBinding(
        profile_id="policy-human-allocation-scheme-a",
        cycle_kind="allocation_proposal_execution",
        builder=lambda assembly: (
            calls.append(assembly)
            or SimHILApplicationControlProfile(
                profile_id="policy-human-allocation-scheme-a",
                session=assembly.compatibility_session,
                control_cycle_port=_Port("allocation_proposal_execution"),
            )
        ),
    )

    profile = select_sim_hil_application_control_profile(
        selected.profile_id,
        assembly=_assembly(),
        additional_bindings=(selected,),
    )

    assert profile.control_cycle_port.spec.cycle_kind == (
        "allocation_proposal_execution"
    )
    assert len(calls) == 1

    never_called = SimHILControlProfileBinding(
        profile_id="must-remain-inert",
        cycle_kind="other",
        builder=lambda _assembly: pytest.fail("unselected profile was constructed"),
    )
    compatibility = select_sim_hil_application_control_profile(
        SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
        assembly=_assembly(),
        additional_bindings=(never_called,),
    )
    assert isinstance(
        compatibility.control_cycle_port,
        FixedAlphaCompatibilityControlCyclePort,
    )


def test_profile_selection_rejects_unknown_duplicate_and_behavior_mismatch() -> None:
    with pytest.raises(ValueError, match="No Sim-HIL application control binding"):
        select_sim_hil_application_control_profile(
            "unknown-profile",
            assembly=_assembly(),
        )

    duplicate = SimHILControlProfileBinding(
        profile_id=SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
        cycle_kind="other",
        builder=lambda assembly: SimHILApplicationControlProfile(
            profile_id=SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
            session=assembly.compatibility_session,
            control_cycle_port=_Port("other"),
        ),
    )
    with pytest.raises(ValueError, match="profile IDs must be unique"):
        select_sim_hil_application_control_profile(
            SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
            assembly=_assembly(),
            additional_bindings=(duplicate,),
        )

    mismatch = SimHILControlProfileBinding(
        profile_id="mismatch-profile",
        cycle_kind="declared-kind",
        builder=lambda assembly: SimHILApplicationControlProfile(
            profile_id="mismatch-profile",
            session=assembly.compatibility_session,
            control_cycle_port=_Port("actual-kind"),
        ),
    )
    with pytest.raises(ValueError, match="differs from its profile binding"):
        select_sim_hil_application_control_profile(
            mismatch.profile_id,
            assembly=_assembly(),
            additional_bindings=(mismatch,),
        )


def test_factory_rejects_unbound_profile_before_any_source_is_built() -> None:
    source_builder_calls = []
    config = MujocoSimHILApplicationConfig(
        enabled=True,
        control_profile_id="unbound-profile",
    )

    with pytest.raises(ValueError, match="No Sim-HIL application control binding"):
        build_application_graph(
            config,
            human_source_builder=lambda **kwargs: source_builder_calls.append(kwargs),
        )

    assert source_builder_calls == []
