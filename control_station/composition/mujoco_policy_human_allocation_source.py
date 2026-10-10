"""Source sequencing for the MuJoCo policy/human allocation profile.

The class receives already-configured input publishers.  It owns no policy or
device worker, role derivation, anchor, allocation, authority, safety, target,
or execution semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robot_motion.deploy import PolicyActionEmission
from teleop_runtime.action.human_proposal import HumanCommandEmission

from .mujoco_allocation_control_cycle import (
    MujocoAllocationSourcePublication,
)

MujocoPolicyAllocationInputPublisher = Callable[
    [PolicyActionEmission, float],
    Any,
]
MujocoHumanAllocationInputPublisher = Callable[
    [HumanCommandEmission, float, float],
    Any,
]


def _accepted(submission: Any) -> bool:
    accepted = getattr(submission, "accepted", None)
    if accepted is not None:
        return bool(accepted)
    return str(getattr(submission, "status", "")) == "batch_admitted"


def _rejection_detail(submission: Any) -> str:
    if submission is None:
        return "missing_submission"
    error = str(getattr(submission, "error", "") or "").strip()
    status = str(getattr(submission, "status", "") or "").strip()
    input_cycle = getattr(submission, "input_cycle", None)
    cycle_status = str(getattr(input_cycle, "status", "") or "").strip()
    if cycle_status == "batch_rejected":
        publication = getattr(input_cycle, "publication", None)
        receipt = getattr(publication, "receipt", None)
        rejection_code = str(getattr(receipt, "rejection_code", "") or "").strip()
        rejection_message = str(getattr(receipt, "message", "") or "").strip()
        if rejection_code:
            details = ":".join(
                part for part in (rejection_code, rejection_message) if part
            )
            return f"{error or cycle_status}:{details}"
    return error or cycle_status or status or type(submission).__name__


class MujocoPolicyHumanAllocationSourcePublisher:
    """Publish one source cut without inferring roles or fallback."""

    def __init__(
        self,
        *,
        human_source: Any,
        publish_policy_input: MujocoPolicyAllocationInputPublisher,
        publish_human_input: MujocoHumanAllocationInputPublisher,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        if not callable(getattr(human_source, "step", None)):
            raise TypeError("human_source must expose step(now=...).")
        if not callable(publish_policy_input):
            raise TypeError("publish_policy_input must be callable.")
        if not callable(publish_human_input):
            raise TypeError("publish_human_input must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        domain = str(clock_domain_id).strip()
        if not domain:
            raise ValueError("clock_domain_id must be non-empty.")
        self.human_source = human_source
        self.publish_policy_input = publish_policy_input
        self.publish_human_input = publish_human_input
        self.clock = clock
        self.clock_domain_id = domain

    def __call__(
        self,
        now: float,
        policy_emission: PolicyActionEmission | None,
        human_emission: Any,
        poll_human: bool,
        target_effective_at: float,
    ) -> MujocoAllocationSourcePublication:
        timestamp = float(now)
        target = float(target_effective_at)
        if human_emission is not None and poll_human:
            raise ValueError(
                "A supplied activation emission and human polling are mutually "
                "exclusive."
            )
        human_step = None
        selected_human = human_emission
        if poll_human:
            human_step = self.human_source.step(now=timestamp)
            selected_human = getattr(human_step, "emission", None)
        if selected_human is not None and not isinstance(
            selected_human,
            HumanCommandEmission,
        ):
            raise TypeError("Human source must provide HumanCommandEmission.")

        # Polling a concrete TeleopSession creates the human emission inside
        # ``step``.  Close the receiver cut after that call: using the
        # pre-poll ``now`` as source_available_at can otherwise make an
        # immediate rebase appear to predate the exact source sample it
        # captures.  This is receiver timing only; source timestamps remain
        # unchanged and no action value is reconstructed here.
        observed_at = max(timestamp, float(self.clock()))

        policy_submission = None
        if policy_emission is not None:
            if not isinstance(policy_emission, PolicyActionEmission):
                raise TypeError("Policy source must provide PolicyActionEmission.")
            policy_submission = self.publish_policy_input(
                policy_emission,
                observed_at,
            )
            observed_at = max(observed_at, float(self.clock()))
        human_submission = None
        if selected_human is not None:
            human_submission = self.publish_human_input(
                selected_human,
                observed_at,
                target,
            )
            observed_at = max(observed_at, float(self.clock()))

        # Receiver admission can take time too. The returned cut must include
        # both completed receipts; it must not reuse the pre-admission cut and
        # accidentally select an older input or allocation context. Source
        # timestamps and their support deadlines are unchanged.

        # A non-null supplied human emission marks activation prime. Both
        # required contributors must cross their receiver boundary before the
        # application may enter run_cycle(). Ordinary cycles remain tolerant
        # of a source producing no new sample; TTL/required-role selection and
        # the lower execution owner decide the response.
        if human_emission is not None:
            if policy_submission is None or not _accepted(policy_submission):
                raise RuntimeError(
                    "Fresh policy activation input was not admitted: "
                    f"{_rejection_detail(policy_submission)}."
                )
            if human_submission is None or not _accepted(human_submission):
                raise RuntimeError(
                    "Fresh human activation input was not admitted: "
                    f"{_rejection_detail(human_submission)}."
                )

        return MujocoAllocationSourcePublication(
            observed_at=observed_at,
            observed_clock_domain_id=self.clock_domain_id,
            human_step=human_step,
            human_submission=human_submission,
            policy_submission=policy_submission,
            input_publication=getattr(human_submission, "publication", None),
        )


__all__ = [
    "MujocoHumanAllocationInputPublisher",
    "MujocoPolicyAllocationInputPublisher",
    "MujocoPolicyHumanAllocationSourcePublisher",
]
