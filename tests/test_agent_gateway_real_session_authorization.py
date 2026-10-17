from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real import motion_gateway_session, session_authorization


class _Manifest:
    runtime_session_id = "real-session:absolute-pose-test"
    semantic_content_hash = "route-manifest:absolute-pose-test"
    control_domain_id = "real-control:absolute-pose-test"
    target_binding_id = "real-target:absolute-pose-test"

    def route(self, skill_id, skill_version, route_id=None, route_version=None):
        del skill_id, skill_version, route_id, route_version
        return SimpleNamespace(route=SimpleNamespace())


class _Gateway:
    def __init__(self) -> None:
        self.contract_bundle = SimpleNamespace(route_manifest=_Manifest())
        self.service = SimpleNamespace(access_policy=SimpleNamespace())


def _request(*, component_id: str = "right_arm") -> SimpleNamespace:
    return SimpleNamespace(
        client_id="rpent.dexteleop",
        skill_execution_grant_id=None,
        skill_id="motion.move_tcp_to_pose",
        skill_version="1",
        route_id="real.rm65.agent_skill_tcp_pose_target",
        route_version="1",
        parameter_profile_id="real-tool-tcp-absolute-grasp-pose-v1",
        parameter_profile_hash="profile:absolute-pose-test",
        parameters={
            "pose_targets_by_component": {
                component_id: [0.25, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0]
            }
        },
    )


def test_absolute_tcp_pose_request_receives_a_request_bound_grant(monkeypatch) -> None:
    gateway = _Gateway()
    issued: list[object] = []

    monkeypatch.setattr(
        motion_gateway_session,
        "RealAgentPrimitiveGatewaySession",
        _Gateway,
    )

    def issue(**kwargs):
        issued.append(kwargs)
        return SimpleNamespace(grant_id="agent-skill-grant:absolute-pose-test")

    monkeypatch.setattr(session_authorization, "issue_skill_execution_grant", issue)
    authorization = session_authorization.RealAgentPrimitiveSessionAuthorization(
        gateway_session=gateway,
        side="right",
        action=session_authorization.REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
        client_id="rpent.dexteleop",
        confirmation_token=(
            session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
        ),
        clock=lambda: 10.0,
    )
    request = _request()

    grant = authorization.grant_resolver(request)

    assert grant is not None
    assert len(issued) == 1
    assert issued[0]["request"] is request
    assert authorization.grants_issued == 1


def test_absolute_tcp_pose_grant_rejects_wrong_side_and_unlisted_action(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        motion_gateway_session,
        "RealAgentPrimitiveGatewaySession",
        _Gateway,
    )
    monkeypatch.setattr(
        session_authorization,
        "issue_skill_execution_grant",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("rejected requests must not mint a Grant")
        ),
    )
    gateway = _Gateway()
    wrong_side_authorization = (
        session_authorization.RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="left",
            action=session_authorization.REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
            client_id="rpent.dexteleop",
            confirmation_token=(
                session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
            ),
        )
    )
    unlisted_authorization = (
        session_authorization.RealAgentPrimitiveSessionAuthorization(
            gateway_session=gateway,
            side="right",
            action="forward",
            client_id="rpent.dexteleop",
            confirmation_token=(
                session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
            ),
        )
    )

    assert wrong_side_authorization.grant_resolver(_request()) is None
    assert unlisted_authorization.grant_resolver(_request()) is None


def test_authorization_still_requires_the_concrete_gateway_owner():
    with pytest.raises(TypeError, match="gateway_session must be typed"):
        session_authorization.RealAgentPrimitiveSessionAuthorization(
            gateway_session=SimpleNamespace(), side="right", action="up",
            client_id="operator:test",
            confirmation_token=session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        )
