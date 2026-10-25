from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.real import motion_gateway_session, session_authorization
from agent_gateway.skills.motion.definition import ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1


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
        self.contract_bundle = SimpleNamespace(route_manifest=_Manifest(),
            attachment=SimpleNamespace(component_ids=("left_arm", "right_arm")))
        self.service = SimpleNamespace(access_policy=SimpleNamespace())


def _request(*, component_id: str = "right_arm", action: str = "move_tcp_to_pose") -> SimpleNamespace:
    return SimpleNamespace(
        client_id="rpent.dexteleop",
        skill_execution_grant_id=None,
        skill_id="motion." + action,
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


def test_dual_pose_authority_mints_only_separate_single_arm_grants(monkeypatch):
    monkeypatch.setattr(motion_gateway_session, "RealAgentPrimitiveGatewaySession", _Gateway)
    issued = []
    monkeypatch.setattr(session_authorization, "issue_skill_execution_grant",
        lambda **kwargs: issued.append(kwargs) or SimpleNamespace(grant_id=str(len(issued))))
    authority = session_authorization.RealAgentPrimitiveSessionAuthorization(
        gateway_session=_Gateway(), side="right", authorized_motion_sides=("left", "right"),
        action="move_tcp_to_pose_safe", client_id="rpent.dexteleop",
        confirmation_token=session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)
    for arm in ("left", "right"):
        assert authority.grant_resolver(_request(component_id=arm+"_arm", action="move_tcp_to_pose_safe"))
    both = _request(action="move_tcp_to_pose_safe")
    both.parameters["pose_targets_by_component"]["left_arm"] = [0, 0, .3, 0, 0, 0, 1]
    assert authority.grant_resolver(both) is None
    assert len(issued) == 2


@pytest.mark.parametrize("sides", [(), ("left",), ("right", "other")])
def test_dual_authority_rejects_invalid_scope(monkeypatch, sides):
    monkeypatch.setattr(motion_gateway_session, "RealAgentPrimitiveGatewaySession", _Gateway)
    with pytest.raises(ValueError, match="authorized_motion_sides"):
        session_authorization.RealAgentPrimitiveSessionAuthorization(
            gateway_session=_Gateway(), side="right", authorized_motion_sides=sides,
            action="move_tcp_to_pose_safe", client_id="rpent.dexteleop",
            confirmation_token=session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN)


@pytest.mark.parametrize("action", ["move_tcp_to_pose", "move_tcp_to_pose_safe"])
def test_absolute_tcp_pose_request_receives_a_request_bound_grant(monkeypatch, action) -> None:
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
        action=action,
        client_id="rpent.dexteleop",
        confirmation_token=(
            session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
        ),
        clock=lambda: 10.0,
    )
    request = _request(action=action)

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
            action="approach_grasp_pose",
            client_id="rpent.dexteleop",
            confirmation_token=(
                session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
            ),
        )
    )

    assert wrong_side_authorization.grant_resolver(_request()) is None
    assert unlisted_authorization.grant_resolver(_request()) is None


def test_legacy_pose_grant_cannot_authorize_safe_move(monkeypatch):
    monkeypatch.setattr(motion_gateway_session, "RealAgentPrimitiveGatewaySession", _Gateway)
    monkeypatch.setattr(session_authorization, "issue_skill_execution_grant",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not issue a grant")))
    authorization = session_authorization.RealAgentPrimitiveSessionAuthorization(
        gateway_session=_Gateway(), side="right", action="move_tcp_to_pose",
        client_id="rpent.dexteleop",
        confirmation_token=session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    )
    assert authorization.grant_resolver(_request(action="move_tcp_to_pose_safe")) is None


def test_authorization_still_requires_the_concrete_gateway_owner():
    with pytest.raises(TypeError, match="gateway_session must be typed"):
        session_authorization.RealAgentPrimitiveSessionAuthorization(
            gateway_session=SimpleNamespace(), side="right", action="move_tcp_to_pose",
            client_id="operator:test",
            confirmation_token=session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
        )


@pytest.mark.parametrize("skill_id", sorted(ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1))
def test_archived_real_motion_selection_cannot_authorize_or_mint_a_grant(monkeypatch, skill_id):
    monkeypatch.setattr(motion_gateway_session, "RealAgentPrimitiveGatewaySession", _Gateway)
    gateway = _Gateway()
    kwargs = {
        "gateway_session": gateway,
        "side": "right",
        "client_id": "rpent.dexteleop",
        "confirmation_token": session_authorization.REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
    }
    action = skill_id.removeprefix("motion.step_tcp_").removeprefix("motion.set_tcp_")
    for selected in (action, skill_id):
        with pytest.raises(ValueError, match="Archived debug motion"):
            session_authorization.RealAgentPrimitiveSessionAuthorization(action=selected, **kwargs)
    authorization = session_authorization.RealAgentPrimitiveSessionAuthorization(
        action="move_tcp_to_pose", **kwargs,
    )
    request = _request()
    request.skill_id = skill_id
    request.parameters = {"component_id": "right_arm", "direction": "forward"}
    assert authorization.grant_resolver(request) is None
    assert authorization.grants_issued == 0
