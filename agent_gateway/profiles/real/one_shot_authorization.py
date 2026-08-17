"""One-use operator authorization shared by real primitive application hosts."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.access_policy import (
    SkillExecutionGrant,
    issue_skill_execution_grant,
)
from agent_gateway.skills.motion.definition import (
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .motion_gateway_session import RealAgentPrimitiveGatewaySession


REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN = "AUTHORIZE_ONE_REAL_TCP_PRIMITIVE"
REAL_AGENT_PRIMITIVE_ACTIONS = (
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
    "level_forward",
    "level_forward_left_30",
    "level_forward_right_30",
)
_FIXED_SKILL_ID_BY_ACTION = {
    "level_forward": TCP_LEVEL_FORWARD_SKILL_ID_V1,
    "level_forward_left_30": TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    "level_forward_right_30": TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
}
_ACTION_BY_FIXED_SKILL_ID = {
    skill_id: action for action, skill_id in _FIXED_SKILL_ID_BY_ACTION.items()
}


class RealAgentPrimitiveOneShotAuthorization:
    """Issue at most one request-bound Grant from an approved action set."""

    def __init__(
        self,
        *,
        gateway_session: RealAgentPrimitiveGatewaySession,
        side: str,
        action: str | None = None,
        authorized_actions: tuple[str, ...] | None = None,
        client_id: str,
        confirmation_token: str,
        grant_lifetime_sec: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(gateway_session, RealAgentPrimitiveGatewaySession):
            raise TypeError("gateway_session must be typed.")
        normalized_side = str(side).strip()
        if authorized_actions is None:
            if action is None:
                raise ValueError("action or authorized_actions is required.")
            requested_actions = (action,)
        else:
            if action is not None:
                raise ValueError("action and authorized_actions are mutually exclusive.")
            requested_actions = tuple(authorized_actions)
        normalized_actions = tuple(
            sorted({str(item).strip() for item in requested_actions})
        )
        normalized_client_id = str(client_id).strip()
        if normalized_side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        if not normalized_actions or any(
            item not in REAL_AGENT_PRIMITIVE_ACTIONS for item in normalized_actions
        ):
            raise ValueError("authorized_actions contains an unreviewed primitive.")
        if not normalized_client_id:
            raise ValueError("client_id must be non-empty.")
        if str(confirmation_token) != REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN:
            raise ValueError("exact real primitive confirmation token is required.")
        lifetime = float(grant_lifetime_sec)
        if not 0.0 < lifetime <= 30.0:
            raise ValueError("grant_lifetime_sec must be in (0, 30].")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.gateway_session = gateway_session
        self.side = normalized_side
        self.authorized_actions = normalized_actions
        self.action = normalized_actions[0] if len(normalized_actions) == 1 else None
        self.client_id = normalized_client_id
        self.grant_lifetime_sec = lifetime
        self._clock = clock
        self._lock = threading.Lock()
        self._grant_issued = False

    @property
    def expected_skill_id(self) -> str:
        if self.action is None:
            raise RuntimeError("expected_skill_id is undefined for an action set.")
        return _FIXED_SKILL_ID_BY_ACTION.get(
            self.action,
            "motion.step_tcp_cardinal",
        )

    @property
    def grant_issued(self) -> bool:
        with self._lock:
            return self._grant_issued

    def grant_resolver(
        self,
        request: AgentSkillRequest,
    ) -> SkillExecutionGrant | None:
        """Resolve one exact request; every mismatch remains unauthorized."""

        if request.client_id != self.client_id:
            return None
        requested_action = (
            request.parameters.get("direction")
            if request.skill_id == "motion.step_tcp_cardinal"
            else _ACTION_BY_FIXED_SKILL_ID.get(request.skill_id)
        )
        if requested_action not in self.authorized_actions:
            return None
        if request.parameters.get("component_id") != f"{self.side}_arm":
            return None
        if requested_action not in REAL_AGENT_PRIMITIVE_ACTIONS:
            return None
        with self._lock:
            if self._grant_issued:
                return None
            now = float(self._clock())
            route = self.gateway_session.contract_bundle.route_manifest.route(
                request.skill_id,
                request.skill_version,
                request.route_id,
                request.route_version,
            ).route
            grant = issue_skill_execution_grant(
                access_policy=self.gateway_session.service.access_policy,
                manifest=self.gateway_session.contract_bundle.route_manifest,
                route=route,
                request=request,
                operator_authorization_receipt_id=(
                    "real-agent-primitive-cli-confirmation:"
                    + canonical_content_hash(
                        {
                            "side": self.side,
                            "authorized_actions": self.authorized_actions,
                            "client_id": self.client_id,
                            "confirmation": REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
                        },
                        domain=(
                            "dexteleop.real_agent_motion."
                            "operator_confirmation.v1"
                        ),
                    )
                ),
                client_id=self.client_id,
                parameter_profile_id=request.parameter_profile_id,
                parameter_profile_hash=request.parameter_profile_hash,
                max_invocations=1,
                issued_at=now,
                expires_at=now + self.grant_lifetime_sec,
            )
            self._grant_issued = True
            return grant


__all__ = [
    "REAL_AGENT_PRIMITIVE_ACTIONS",
    "REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN",
    "RealAgentPrimitiveOneShotAuthorization",
]
