from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from control_station.composition.rebot_b601_rm65_mujoco_session import (
    B601RM65MujocoShadowSession,
    B601RM65MujocoShadowSessionProfile,
)
from control_station.composition.rebot_b601_rm65_mujoco_shadow import (
    B601RM65MujocoShadowActivation,
    B601RM65MujocoShadowCycle,
)
from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65PoseProjection,
    B601RM65ResidualStep,
)
from robot_motion.action import ExecutionMeasurementObservation
from teleop_runtime.action.human_residual import HumanResidualPoseAnchor

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


class _Source:
    def __init__(self, projections: list[object]) -> None:
        self.projections = list(projections)
        self.anchor = Mock(spec=HumanResidualPoseAnchor)
        self.residual = Mock(spec=B601RM65ResidualStep)
        self.poll_count = 0
        self.capture_calls: list[dict[str, object]] = []
        self.adapt_calls: list[dict[str, object]] = []

    def poll_pose(self):
        self.poll_count += 1
        return self.projections.pop(0)

    def capture_anchor(self, projection, **kwargs):
        self.capture_calls.append({"projection": projection, **kwargs})
        return self.anchor

    def adapt(self, projection, **kwargs):
        self.adapt_calls.append({"projection": projection, **kwargs})
        return self.residual


class _Shadow:
    def __init__(self, *, activation_status: str = "activated") -> None:
        self.profile = SimpleNamespace(enabled=True, send_actions=False)
        self.runtime = SimpleNamespace(
            config=SimpleNamespace(clock_domain_id="control.monotonic")
        )
        self.contracts = SimpleNamespace(
            composition_spec=SimpleNamespace(
                required_roles=("target_baseline", "human_residual")
            )
        )
        self.activation_statuses = [activation_status]
        self.cycle_statuses = ["evaluated"]
        self.activate_calls: list[dict[str, object]] = []
        self.evaluate_calls: list[dict[str, object]] = []

    def activate(self, scope, observation, **kwargs):
        self.activate_calls.append(
            {"scope": scope, "observation": observation, **kwargs}
        )
        result = Mock(spec=B601RM65MujocoShadowActivation)
        result.status = (
            self.activation_statuses.pop(0)
            if self.activation_statuses
            else "activated"
        )
        result.activation_scope = scope
        return result

    def evaluate(self, residual, **kwargs):
        self.evaluate_calls.append({"residual": residual, **kwargs})
        result = Mock(spec=B601RM65MujocoShadowCycle)
        result.status = (
            self.cycle_statuses.pop(0)
            if self.cycle_statuses
            else "evaluated"
        )
        return result


def _projection(*, admitted: bool = True):
    result = Mock(spec=B601RM65PoseProjection)
    result.admitted = admitted
    result.rejection_reason = None if admitted else "synthetic_stale_source"
    return result


def _observation(*, suffix: str = "1") -> ExecutionMeasurementObservation:
    return ExecutionMeasurementObservation(
        measurement_id=f"rm65-measured-pose:{suffix}",
        source_measurement_hash=f"rm65-measured-qpos:{suffix}",
        control_domain_id="rm65-sim",
        measured_at=9.98,
        available_at=9.99,
        clock_domain_id="control.monotonic",
        value_by_component={
            "right_arm": (0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0)
        },
        representation_by_component={"right_arm": "cartesian_pose"},
        schema_hash_by_component={"right_arm": "rm65-pose-schema"},
        observation_binding_id="rm65-pose-observation",
        observation_plan_hash="rm65-pose-plan",
    )


def _profile() -> B601RM65MujocoShadowSessionProfile:
    return B601RM65MujocoShadowSessionProfile(
        profile_id="b601-rm65-shadow-session",
        operation_spec_id="b601-rm65-shadow",
        operation_spec_hash="b601-rm65-shadow-operation-hash",
        clock_domain_id="control.monotonic",
        effective_horizon_sec=0.2,
        enabled=True,
    )


def _session(source, shadow, clock, *, first_session_epoch: int = 1):
    return B601RM65MujocoShadowSession(
        profile=_profile(),
        residual_source=source,
        shadow_controller=shadow,
        clock=clock,
        first_session_epoch=first_session_epoch,
    )


def test_default_off_does_not_evaluate_dependency_suppliers() -> None:
    session = B601RM65MujocoShadowSession(
        profile=B601RM65MujocoShadowSessionProfile()
    )

    activation = session.activate(None)  # type: ignore[arg-type]
    cycle = session.step(target_effective_at=10.0)

    assert activation.status == "disabled"
    assert cycle.status == "disabled"
    assert session.snapshot.state == "disabled"
    assert not activation.command_dispatched
    assert not cycle.command_dispatched


def test_activation_captures_both_anchors_and_step_uses_current_sample() -> None:
    first = _projection()
    current = _projection()
    source = _Source([first, current])
    shadow = _Shadow()
    session = _session(source, shadow, _Clock(10.0, 10.05), first_session_epoch=3)

    activation = session.activate(_observation(), context_ids=("activation",))
    cycle = session.step(
        target_effective_at=10.08,
        context_ids=("cycle",),
    )

    assert activation.status == "activated"
    assert activation.snapshot.state == "active"
    assert activation.snapshot.session_epoch == 3
    assert activation.snapshot.anchor_generation == 0
    assert source.capture_calls == [
        {
            "projection": first,
            "anchor_generation": 0,
            "captured_at": 10.0,
            "captured_clock_domain_id": "control.monotonic",
        }
    ]
    assert shadow.activate_calls[0]["effective_until"] == pytest.approx(10.2)
    assert cycle.status == "evaluated"
    assert cycle.master_projection is current
    assert source.adapt_calls[0]["anchor"] is source.anchor
    assert source.adapt_calls[0]["adapted_at"] == 10.05
    assert shadow.evaluate_calls[0]["residual"] is source.residual
    assert shadow.evaluate_calls[0]["effective_from"] == 10.05
    assert shadow.evaluate_calls[0]["effective_until"] == pytest.approx(10.25)
    assert shadow.evaluate_calls[0]["target_effective_at"] == 10.08
    assert not cycle.authoritative
    assert not cycle.promotable
    assert not cycle.command_dispatched


def test_activation_source_rejection_does_not_reach_target() -> None:
    source = _Source([_projection(admitted=False)])
    shadow = _Shadow()
    session = _session(source, shadow, _Clock())

    result = session.activate(_observation())

    assert result.status == "source_rejected"
    assert result.snapshot.state == "inactive"
    assert "source_rejected" in result.snapshot.reason
    assert source.capture_calls == []
    assert shadow.activate_calls == []


def test_target_rejection_does_not_leave_session_active() -> None:
    source = _Source([_projection()])
    shadow = _Shadow(activation_status="baseline_rejected")
    session = _session(source, shadow, _Clock(10.0))

    result = session.activate(_observation())

    assert result.status == "target_rejected"
    assert result.snapshot.state == "inactive"
    assert "target_rejected" in result.snapshot.reason
    assert len(source.capture_calls) == 1
    assert len(shadow.activate_calls) == 1


def test_clutch_withdraws_without_poll_and_failed_release_stays_clutched() -> None:
    initial = _projection()
    rejected_recenter = _projection(admitted=False)
    successful_recenter = _projection()
    source = _Source([initial, rejected_recenter, successful_recenter])
    shadow = _Shadow()
    session = _session(source, shadow, _Clock(10.0, 11.0))
    assert session.activate(_observation()).status == "activated"

    clutched = session.engage_clutch()
    skipped = session.step(target_effective_at=10.1)
    rejected = session.release_clutch(_observation(suffix="2"))

    assert clutched.state == "clutched"
    assert skipped.status == "clutched"
    assert source.poll_count == 2
    assert rejected.status == "source_rejected"
    assert rejected.snapshot.state == "clutched"
    assert len(shadow.activate_calls) == 1

    resumed = session.release_clutch(_observation(suffix="3"))

    assert resumed.status == "activated"
    assert resumed.snapshot.state == "active"
    assert resumed.snapshot.session_epoch == 2
    assert resumed.snapshot.anchor_generation == 1
    assert source.capture_calls[-1]["projection"] is successful_recenter
    assert len(shadow.activate_calls) == 2


def test_current_source_rejection_is_cycle_local_without_cached_fallback() -> None:
    source = _Source([_projection(), _projection(admitted=False), _projection()])
    shadow = _Shadow()
    shadow.cycle_statuses = ["source_rejected", "evaluated"]
    session = _session(source, shadow, _Clock(10.0, 10.05, 10.10))
    session.activate(_observation())

    rejected = session.step(target_effective_at=10.08)
    resumed = session.step(target_effective_at=10.12)

    assert rejected.status == "source_rejected"
    assert rejected.snapshot.state == "active"
    assert resumed.status == "evaluated"
    assert len(shadow.evaluate_calls) == 2
    assert shadow.evaluate_calls[0]["residual"] is source.residual
    assert shadow.evaluate_calls[1]["residual"] is source.residual


def test_step_exception_fails_closed_and_requires_fresh_activation() -> None:
    source = _Source([_projection(), _projection()])
    shadow = _Shadow()
    session = _session(source, shadow, _Clock(10.0, 10.1))
    session.activate(_observation())

    with pytest.raises(ValueError, match="cannot precede"):
        session.step(target_effective_at=10.05)

    assert session.snapshot.state == "inactive"
    assert session.snapshot.reason == "shadow_step_exception"
    assert shadow.evaluate_calls == []
    assert session.step(target_effective_at=10.2).status == "inactive"


def test_deactivate_does_not_close_runtime_or_source() -> None:
    source = _Source([_projection()])
    shadow = _Shadow()
    shadow.close = Mock()
    source.close = Mock()
    session = _session(source, shadow, _Clock(10.0))
    session.activate(_observation())

    snapshot = session.deactivate(reason="operator_finished_shadow")

    assert snapshot.state == "inactive"
    assert snapshot.reason == "operator_finished_shadow"
    shadow.close.assert_not_called()
    source.close.assert_not_called()


@pytest.mark.parametrize("horizon", [0.0, -0.1, float("nan")])
def test_profile_rejects_invalid_effective_horizon(horizon: float) -> None:
    with pytest.raises(ValueError):
        B601RM65MujocoShadowSessionProfile(effective_horizon_sec=horizon)


def test_enabled_session_requires_non_dispatching_complete_ports() -> None:
    source = _Source([_projection()])
    shadow = _Shadow()
    shadow.profile.send_actions = True

    with pytest.raises(ValueError, match="non-dispatching"):
        _session(source, shadow, _Clock(10.0))


def test_enabled_session_requires_the_runtime_clock_domain() -> None:
    source = _Source([_projection()])
    shadow = _Shadow()
    shadow.runtime.config.clock_domain_id = "different.monotonic"

    with pytest.raises(ValueError, match="clock domains differ"):
        _session(source, shadow, _Clock(10.0))


def test_profile_semantic_hash_covers_the_clock_contract() -> None:
    baseline = _profile()
    changed = B601RM65MujocoShadowSessionProfile(
        profile_id=baseline.profile_id,
        operation_spec_id=baseline.operation_spec_id,
        operation_spec_hash=baseline.operation_spec_hash,
        clock_domain_id="other.monotonic",
        effective_horizon_sec=baseline.effective_horizon_sec,
        enabled=True,
    )

    assert baseline.semantic_content_hash != changed.semantic_content_hash


def test_unexpected_target_activation_status_fails_closed() -> None:
    source = _Source([_projection()])
    shadow = _Shadow(activation_status="unexpected")
    session = _session(source, shadow, _Clock(10.0))

    with pytest.raises(RuntimeError, match="invalid activation status"):
        session.activate(_observation())

    assert session.snapshot.state == "inactive"
    assert session.snapshot.reason == "activation_exception"


def test_untyped_target_activation_fails_closed() -> None:
    source = _Source([_projection()])
    shadow = _Shadow()
    shadow.activate = Mock(return_value=object())
    session = _session(source, shadow, _Clock(10.0))

    with pytest.raises(TypeError, match="untyped activation"):
        session.activate(_observation())

    assert session.snapshot.state == "inactive"
    assert session.snapshot.reason == "activation_exception"


def test_untyped_shadow_cycle_fails_closed() -> None:
    source = _Source([_projection(), _projection()])
    shadow = _Shadow()
    session = _session(source, shadow, _Clock(10.0, 10.05))
    session.activate(_observation())
    shadow.evaluate = Mock(return_value=object())

    with pytest.raises(TypeError, match="untyped cycle"):
        session.step(target_effective_at=10.08)

    assert session.snapshot.state == "inactive"
    assert session.snapshot.reason == "shadow_step_exception"


def test_module_has_no_hardware_or_runtime_lifecycle_and_dispatch_calls() -> None:
    source = (
        Path(__file__).parents[1]
        / "control_station"
        / "composition"
        / "rebot_b601_rm65_mujoco_session.py"
    ).read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source
    assert ".start(" not in source
    assert ".close(" not in source
    assert ".dispatch(" not in source
    assert "compose_and_submit" not in source
    assert "submit_evaluated_composition" not in source
