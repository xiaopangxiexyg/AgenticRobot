"""Persistent operator session authorization for real primitive application hosts."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping

from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.access_policy import (
    SkillExecutionGrant,
    issue_skill_execution_grant,
)
from agent_gateway.skills.motion.definition import (
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .motion_gateway_session import RealAgentPrimitiveGatewaySession

REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN = "AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES"
# Compatibility name used by the existing composition API. Its value is the
# persistent-session token; the retired one-shot token is intentionally invalid.
REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN = REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION = "move_tcp_to_pose"
REAL_AGENT_GRASP_APPROACH_ACTION = "approach_grasp_pose"
REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION = "execute_grasp_approach_plan"
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
    REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
    REAL_AGENT_GRASP_APPROACH_ACTION,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION,
)
_FIXED_SKILL_ID_BY_ACTION = {
    **dict(FIXED_CARDINAL_TCP_STEP_SKILL_ID_BY_DIRECTION_V1),
    "level_forward": TCP_LEVEL_FORWARD_SKILL_ID_V1,
    "level_forward_left_30": TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    "level_forward_right_30": TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION: TCP_POSE_TARGET_SKILL_ID_V1,
    REAL_AGENT_GRASP_APPROACH_ACTION: GRASP_APPROACH_SKILL_ID_V1,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION: (
        EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1
    ),
}
_ACTION_BY_FIXED_SKILL_ID = {
    skill_id: action for action, skill_id in _FIXED_SKILL_ID_BY_ACTION.items()
}


def _requested_action_and_component_ids(
    request: AgentSkillRequest,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve only the operator authorization class and selected components.

    Absolute TCP targets intentionally use ``pose_targets_by_component`` while
    the fixed primitives use ``component_id``.  Keeping this distinction here
    prevents a published absolute-pose Route from becoming permanently
    ungrantable merely because its payload shape is not a cardinal primitive.
    Target values, frame identities, profile hashes, safety envelopes, and all
    lower admission remain validated by their existing owners.
    """

    parameters = request.parameters
    fixed_cardinal_action = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1.get(
        request.skill_id
    )
    if (
        fixed_cardinal_action is not None
        and parameters.get("direction") != fixed_cardinal_action
    ):
        return None, ()
    if request.skill_id in {
        TCP_POSE_TARGET_SKILL_ID_V1,
        GRASP_APPROACH_SKILL_ID_V1,
    }:
        targets = parameters.get("pose_targets_by_component")
        if not isinstance(targets, Mapping):
            return _ACTION_BY_FIXED_SKILL_ID[request.skill_id], ()
        component_ids = tuple(
            sorted(
                key
                for key in targets
                if isinstance(key, str) and key.strip() == key and key
            )
        )
        if len(component_ids) != len(targets):
            return _ACTION_BY_FIXED_SKILL_ID[request.skill_id], ()
        return _ACTION_BY_FIXED_SKILL_ID[request.skill_id], component_ids
    requested_action = (
        parameters.get("direction")
        if request.skill_id == "motion.step_tcp_cardinal"
        else (fixed_cardinal_action or _ACTION_BY_FIXED_SKILL_ID.get(request.skill_id))
    )
    component_id = parameters.get("component_id")
    return (
        requested_action if isinstance(requested_action, str) else None,
        (component_id,) if isinstance(component_id, str) else (),
    )


class RealAgentPrimitiveSessionAuthorization:
    """Mint one request-bound Grant per approved request in one Gateway session."""

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
                raise ValueError(
                    "action and authorized_actions are mutually exclusive."
                )
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
        if str(confirmation_token) != REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN:
            raise ValueError(
                "exact real primitive confirmation token for a persistent "
                "session is required."
            )
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
        self._grants_issued = 0

        manifest = gateway_session.contract_bundle.route_manifest
        self.operator_authorization_receipt_id = (
            "real-agent-primitive-cli-session-confirmation:"
            + canonical_content_hash(
                {
                    "runtime_session_id": manifest.runtime_session_id,
                    "route_manifest_hash": manifest.semantic_content_hash,
                    "control_domain_id": manifest.control_domain_id,
                    "target_binding_id": manifest.target_binding_id,
                    "side": self.side,
                    "authorized_actions": self.authorized_actions,
                    "client_id": self.client_id,
                    "confirmation": REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
                },
                domain=("dexteleop.real_agent_motion.operator_session_confirmation.v1"),
            )
        )

    @property
    def expected_skill_id(self) -> str:
        if self.action is None:
            raise RuntimeError("expected_skill_id is undefined for an action set.")
        return _FIXED_SKILL_ID_BY_ACTION[self.action]

    @property
    def grants_issued(self) -> int:
        with self._lock:
            return self._grants_issued

    @property
    def grant_issued(self) -> bool:
        """Compatibility observation; it is not a one-shot admission gate."""

        return self.grants_issued > 0

    def grant_resolver(
        self,
        request: AgentSkillRequest,
    ) -> SkillExecutionGrant | None:
        """Mint a fresh one-use Grant for each matching pre-grant request."""

        if request.client_id != self.client_id:
            return None
        if request.skill_execution_grant_id is not None:
            return None
        requested_action, component_ids = _requested_action_and_component_ids(request)
        if requested_action not in self.authorized_actions:
            return None
        if component_ids != (f"{self.side}_arm",):
            return None
        if requested_action not in REAL_AGENT_PRIMITIVE_ACTIONS:
            return None

        with self._lock:
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
                    self.operator_authorization_receipt_id
                ),
                client_id=self.client_id,
                parameter_profile_id=request.parameter_profile_id,
                parameter_profile_hash=request.parameter_profile_hash,
                max_invocations=1,
                issued_at=now,
                expires_at=now + self.grant_lifetime_sec,
            )
            self._grants_issued += 1
            return grant


__all__ = [
    "REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION",
    "REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION",
    "REAL_AGENT_GRASP_APPROACH_ACTION",
    "REAL_AGENT_PRIMITIVE_ACTIONS",
    "REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN",
    "REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN",
    "RealAgentPrimitiveSessionAuthorization",
]
