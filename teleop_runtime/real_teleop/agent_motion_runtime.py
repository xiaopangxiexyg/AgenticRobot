"""Default-off composition of the real Agent motion port into RealTeleop."""

from __future__ import annotations

from dataclasses import replace

from .agent_motion_port import (
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from .components import RealTeleopComponents


def _validate_attachment(
    components: RealTeleopComponents,
    *,
    config: RealAgentMotionPortConfig,
) -> None:
    if not isinstance(components, RealTeleopComponents):
        raise TypeError("components must be RealTeleopComponents.")
    if not isinstance(config, RealAgentMotionPortConfig):
        raise TypeError("config must be RealAgentMotionPortConfig.")
    if components.agent_motion_port is not None:
        raise RuntimeError("RealTeleop already has an Agent motion port.")
    target_builder = components.dual_arm_retargeter
    if target_builder is None:
        raise RuntimeError(
            "Real Agent motion requires the existing coupled dual-arm target solver."
        )
    port_sides = set(config.active_sides)
    runtime_sides = set(components.active_sides)
    exact_authority = port_sides == runtime_sides
    exclusive_coupled_subset = bool(
        config.exclusive_command_claim
        and runtime_sides == {"left", "right"}
        and port_sides < runtime_sides
    )
    if not (exact_authority or exclusive_coupled_subset):
        raise ValueError(
            "Agent motion active_sides must exactly match RealTeleop authority, "
            "except that an exclusive coupled dual-arm runtime may authorize "
            "one command side while hard-holding the other."
        )
    session = components.session
    if session.command_override_provider is not None:
        raise RuntimeError(
            "RealTeleop already has a command override provider; ownership is ambiguous."
        )


def build_real_agent_motion_port(
    components: RealTeleopComponents,
    *,
    config: RealAgentMotionPortConfig,
    owner_identity: RealAgentMotionOwnerIdentity | None = None,
) -> RealAgentMotionPort:
    """Validate and build the lower port without mutating the Session."""

    _validate_attachment(components, config=config)
    return RealAgentMotionPort(
        config=config,
        target_builder=components.dual_arm_retargeter,
        owner_identity=owner_identity,
    )


def attach_prepared_real_agent_motion_port(
    components: RealTeleopComponents,
    *,
    port: RealAgentMotionPort,
) -> RealTeleopComponents:
    """Attach a preflighted port as the Session's only command override."""

    if not isinstance(port, RealAgentMotionPort):
        raise TypeError("port must be RealAgentMotionPort.")
    _validate_attachment(components, config=port.config)
    if not port.is_bound_to_target_builder(components.dual_arm_retargeter):
        raise ValueError("Prepared Agent motion port target builder mismatch.")

    session = components.session
    previous_admission = session.command_dispatch_admission

    def combined_admission(command, observation, now):
        if not port.admit_command(command, observation, now):
            return False
        if previous_admission is None:
            return True
        return bool(previous_admission(command, observation, now))

    session.command_override_provider = port.command_for_observation
    session.command_dispatch_admission = combined_admission
    if port.config.exclusive_command_claim:
        port.confirm_exclusive_attachment()
    return replace(components, agent_motion_port=port)


def attach_real_agent_motion_port(
    components: RealTeleopComponents,
    *,
    config: RealAgentMotionPortConfig,
) -> RealTeleopComponents:
    """Attach one lower port without adding a second command/dispatch owner.

    The ordinary component builder never calls this function.  A production
    composition root must opt in with an exact execution-frame/component
    binding before it publishes any real motion Route.
    """

    port = build_real_agent_motion_port(components, config=config)
    return attach_prepared_real_agent_motion_port(components, port=port)


__all__ = [
    "attach_prepared_real_agent_motion_port",
    "attach_real_agent_motion_port",
    "build_real_agent_motion_port",
]
