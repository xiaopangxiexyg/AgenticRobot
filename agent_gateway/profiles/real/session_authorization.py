"""Persistent operator session authorization for real primitive application hosts."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.access_policy import (
    SkillExecutionGrant,
    issue_skill_execution_grant,
)
from agent_gateway.skills.gripper.definition import SET_GRIPPER_STATE_SKILL_ID_V1
from agent_gateway.skills.motion.observed_point import SKILL_ID as OBSERVED_POINT_SKILL_ID
from agent_gateway.skills.motion.definition import (
    ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1,
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

if TYPE_CHECKING:
    from .motion_gateway_session import RealAgentPrimitiveGatewaySession

REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN = "AUTHORIZE_PERSISTENT_REAL_TCP_PRIMITIVES"
# Compatibility name used by the existing composition API. Its value is the
# persistent-session token; the retired one-shot token is intentionally invalid.
REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN = REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION = "move_tcp_to_pose"
REAL_AGENT_SAFE_TCP_POSE_ACTION = "move_tcp_to_pose_safe"
REAL_AGENT_OBSERVED_POINT_ACTION = "move_to_observed_point"
REAL_AGENT_GRASP_APPROACH_ACTION = "approach_grasp_pose"
REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION = "execute_grasp_approach_plan"
REAL_AGENT_PRIMITIVE_ACTIONS = (
    REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
    REAL_AGENT_SAFE_TCP_POSE_ACTION,
    REAL_AGENT_OBSERVED_POINT_ACTION,
    REAL_AGENT_GRASP_APPROACH_ACTION,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION,
    "set_gripper_state",
)
_FIXED_SKILL_ID_BY_ACTION = {
    "set_gripper_state": SET_GRIPPER_STATE_SKILL_ID_V1,
    REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION: TCP_POSE_TARGET_SKILL_ID_V1,
    REAL_AGENT_SAFE_TCP_POSE_ACTION: SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    REAL_AGENT_OBSERVED_POINT_ACTION: OBSERVED_POINT_SKILL_ID,
    REAL_AGENT_GRASP_APPROACH_ACTION: GRASP_APPROACH_SKILL_ID_V1,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION: (
        EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1
    ),
}
_ACTION_BY_FIXED_SKILL_ID = {
    skill_id: action for action, skill_id in _FIXED_SKILL_ID_BY_ACTION.items()
}
_ARCHIVED_PRIMITIVE_ACTIONS = frozenset(
    skill_id.removeprefix("motion.step_tcp_").removeprefix("motion.set_tcp_")
    for skill_id in ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1
)


def validate_real_agent_primitive_action(action: str) -> str:
    """Reject retired aliases and IDs before authorization or resource assembly."""

    if (
        action in _ARCHIVED_PRIMITIVE_ACTIONS
        or action in ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1
    ):
        raise ValueError(f"Archived debug motion action {action!r} is unavailable.")
    if action not in REAL_AGENT_PRIMITIVE_ACTIONS:
        raise ValueError(f"Unreviewed real motion action {action!r}.")
    return action


def _requested_action_and_component_ids(
    request: AgentSkillRequest,
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve only the operator authorization class and selected components.

    Absolute TCP targets use ``pose_targets_by_component`` while retained plan
    execution uses ``component_id``. Archived debug Skills cannot obtain grants.
    Target values, frame identities, profile hashes, safety envelopes, and all
    lower admission remain validated by their existing owners.
    """

    parameters = request.parameters
    if request.skill_id in ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1:
        return None, ()
    if request.skill_id == SET_GRIPPER_STATE_SKILL_ID_V1:
        return "set_gripper_state", (("right_gripper",) if request.skill_version == "1"
                                      else (str(parameters.get("component_id")),))
    if request.skill_id in {
        TCP_POSE_TARGET_SKILL_ID_V1,
        SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
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
    requested_action = _ACTION_BY_FIXED_SKILL_ID.get(request.skill_id)
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
        authorized_gripper_sides: tuple[str, ...] | None = None,
        authorized_motion_sides: tuple[str, ...] | None = None,
        client_id: str,
        confirmation_token: str,
        grant_lifetime_sec: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Only construction needs the concrete owner; clients import action IDs.
        from .motion_gateway_session import RealAgentPrimitiveGatewaySession

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
        if not normalized_actions:
            raise ValueError("authorized_actions contains an unreviewed primitive.")
        for item in normalized_actions:
            validate_real_agent_primitive_action(item)
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
        self.authorized_motion_sides = tuple(sorted(set(
            (normalized_side,) if authorized_motion_sides is None else authorized_motion_sides)))
        if (not self.authorized_motion_sides or normalized_side not in self.authorized_motion_sides
                or not set(self.authorized_motion_sides) <= {"left", "right"}):
            raise ValueError("authorized_motion_sides must contain the primary side and select left/right.")
        if not {arm + "_arm" for arm in self.authorized_motion_sides} <= set(
                gateway_session.contract_bundle.attachment.component_ids):
            raise ValueError("Motion authorization exceeds the bound arm components.")
        self.authorized_actions = normalized_actions
        self.authorized_gripper_sides = tuple(
            sorted(
                set(
                    (normalized_side,)
                    if authorized_gripper_sides is None
                    else authorized_gripper_sides
                )
            )
        )
        if not self.authorized_gripper_sides or any(
            item not in {"left", "right"} for item in self.authorized_gripper_sides
        ):
            raise ValueError("authorized_gripper_sides must select left and/or right.")
        self.action = normalized_actions[0] if len(normalized_actions) == 1 else None
        self.client_id = normalized_client_id
        self.grant_lifetime_sec = lifetime
        self._clock = clock
        self._lock = threading.Lock()
        self._grants_issued = 0
        gripper_contract = None
        if "set_gripper_state" in normalized_actions:
            gripper_contract = gateway_session.contract_bundle.gripper_contract
            if gripper_contract is None:
                raise ValueError("Gripper authorization requires the bound gripper Route.")
            if (
                gripper_contract.deployment_binding.skill_version == "1"
                and (
                    normalized_side != "right"
                    or self.authorized_gripper_sides != ("right",)
                )
            ):
                raise ValueError(
                    "Gripper v1 authorization requires the bound right-gripper Route."
                )

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
                    **({"authorized_gripper_sides": self.authorized_gripper_sides}
                       if gripper_contract is not None and gripper_contract.deployment_binding.skill_version == "2" else {}),
                    "authorized_motion_sides": self.authorized_motion_sides,
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
        if requested_action == "set_gripper_state":
            if len(component_ids) != 1 or component_ids[0] not in {
                side + "_gripper" for side in self.authorized_gripper_sides
            }:
                return None
        elif requested_action in {
            REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION,
            REAL_AGENT_SAFE_TCP_POSE_ACTION,
            REAL_AGENT_OBSERVED_POINT_ACTION,
        }:
            if len(component_ids) != 1 or component_ids[0] not in {
                    arm + "_arm" for arm in self.authorized_motion_sides}:
                return None
        elif component_ids != (f"{self.side}_arm",):
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
    "REAL_AGENT_OBSERVED_POINT_ACTION",
    "REAL_AGENT_ABSOLUTE_TCP_POSE_ACTION",
    "REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ACTION",
    "REAL_AGENT_GRASP_APPROACH_ACTION",
    "REAL_AGENT_PRIMITIVE_ACTIONS",
    "REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN",
    "REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN",
    "REAL_AGENT_SAFE_TCP_POSE_ACTION",
    "RealAgentPrimitiveSessionAuthorization",
    "validate_real_agent_primitive_action",
]
