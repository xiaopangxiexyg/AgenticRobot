"""Bounded signed-command search; no actuator access or physical torque estimate."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HoldingSearchConfig:
    observation_sec: float = 1.0
    observation_timeout_sec: float = 3.0
    movement_rad: float = 0.002
    torque_resolution_nm: float = 0.025
    max_trials: int = 12
    verification_repeats: int = 2

    def __post_init__(self):
        for name in (
            "observation_sec",
            "observation_timeout_sec",
            "movement_rad",
            "torque_resolution_nm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"holding search {name} must be finite and positive")
        if self.observation_timeout_sec < self.observation_sec:
            raise ValueError("holding observation timeout must cover its dwell")
        for name in ("max_trials", "verification_repeats"):
            if (
                type(getattr(self, name)) is not int
                or not 2 <= getattr(self, name) <= 32
            ):
                raise ValueError(f"holding search {name} must be an integer in 2..32")
        if self.verification_repeats > self.max_trials:
            raise ValueError("holding verification must fit the trial budget")


class HoldingForceSearch:
    """Search a usable hold toward zero, not a unique gravity correction.

    Each observation must be made at the same pose after reacquisition and a
    constant, actually dispatched command. Opposite-direction motion provides
    a bracket; a quiet observation provides a feasible value, not a root.
    """

    def __init__(self, config: HoldingSearchConfig, *, seed_nm: float, cap_nm: float,
                 recheck_interior_on_conflict: bool = False):
        if not isinstance(config, HoldingSearchConfig):
            raise TypeError("holding search requires explicit validated settings")
        if type(recheck_interior_on_conflict) is not bool:
            raise TypeError("interior holding recheck must be boolean")
        if (
            any(isinstance(v, bool) or not math.isfinite(v) for v in (seed_nm, cap_nm))
            or not 0 < cap_nm
            or abs(seed_nm) > cap_nm + 1e-9
        ):
            raise ValueError("holding seed must fit the finite command envelope")
        self.config, self.cap_nm = config, cap_nm
        self.candidate_nm = max(-cap_nm, min(cap_nm, seed_nm))
        self.observations = []
        self.transitions = []
        self.state, self.reason = "searching", "verify_arrival_command"
        self._verification = None
        self._recheck_interior_on_conflict = recheck_interior_on_conflict
        self._interior_recheck_start = None

    def record(
        self, *, command_nm: float, outcome: str, evidence: dict,
        requested_command_nm: float | None = None,
    ):
        if self.state != "searching":
            raise ValueError("holding search is terminal")
        if outcome not in {"stationary", "positive_motion", "negative_motion"}:
            raise ValueError("holding search requires a resolved observation")
        if (
            isinstance(command_nm, bool)
            or not math.isfinite(command_nm)
            or abs(command_nm) > self.cap_nm + 1e-9
        ):
            raise ValueError("observed holding command exceeds its envelope")
        if requested_command_nm is not None and (
            isinstance(requested_command_nm, bool)
            or not math.isfinite(requested_command_nm)
            or abs(requested_command_nm) > self.cap_nm + 1e-9
            or abs(requested_command_nm - self.candidate_nm) > 1e-9
        ):
            raise ValueError("holding observation must belong to the current probe")
        changed_plateau = (requested_command_nm is not None
                           and abs(command_nm - requested_command_nm) > 1e-9)
        if changed_plateau and evidence.get("ramp_frozen") is not True:
            raise ValueError("changed holding plateau requires a frozen ramp")
        self.observations.append(
            {**evidence,
             **({"requested_command_nm": requested_command_nm} if requested_command_nm is not None else {}),
             "command_nm": command_nm, "outcome": outcome}
        )
        positive = [
            r["command_nm"]
            for r in self.observations
            if r["outcome"] == "positive_motion"
        ]
        negative = [
            r["command_nm"]
            for r in self.observations
            if r["outcome"] == "negative_motion"
        ]
        stable = [
            r["command_nm"] for r in self.observations if r["outcome"] == "stationary"
        ]
        lower, upper = max(negative, default=-math.inf), min(positive, default=math.inf)
        # Keep every conflicting observation. The coordinated acquisition may
        # verify another observed interior hold, never fit the rejected edge.
        if lower >= upper or any(not lower < v < upper for v in stable):
            eligible = self._eligible_stationary_commands()
            if not self._recheck_interior_on_conflict or lower >= upper or not eligible:
                self.state, self.reason = (
                    "incomplete",
                    "holding_response_not_repeatable_or_monotone",
                )
                return
            if self._interior_recheck_start is None or self._verification not in eligible:
                self._verification = self.candidate_nm = min(eligible, key=abs)
                self._interior_recheck_start = len(self.observations)
                self.reason = "verify_interior_hold_after_inconsistent_boundary"
                self._check_trial_budget()
                return
        if self._recheck_interior_on_conflict:
            stable = self._eligible_stationary_commands()
        if self._verification is not None:
            if outcome != "stationary" or (
                abs(command_nm - self._verification) > 1e-9 and not changed_plateau
            ):
                self.state, self.reason = "incomplete", "holding_verification_changed"
                return
            # The observer can stop a ramp at an actually dispatched level
            # before the requested recheck level. A causal quiet plateau is a
            # new observation, not a failed recheck of an unsent command.
            # Recheck this actual level; do not count the requested level as
            # another observation or relax command equality for completion.
            self._verification = self.candidate_nm = command_nm
            if changed_plateau and self._interior_recheck_start is not None:
                self._interior_recheck_start = len(self.observations)-1
            count = sum(
                r["outcome"] == "stationary"
                and abs(r["command_nm"] - command_nm) <= 1e-9
                for r in self.observations[self._interior_recheck_start or 0:]
            )
            if count >= self.config.verification_repeats:
                self.state, self.reason = (
                    "complete",
                    "verified_usable_hold_not_unique_gravity",
                )
                return
            self.candidate_nm = self._verification
        elif stable:
            self._probe_from_stable(stable, lower, upper)
            if self.state == "complete":
                return
        elif math.isfinite(lower) and math.isfinite(upper):
            self.candidate_nm = (lower + upper) / 2
            if upper - lower <= self.config.torque_resolution_nm:
                self.state, self.reason = (
                    "incomplete",
                    "no_verified_hold_in_resolved_bracket",
                )
                return
        else:
            # Zero and the arrival seed need not straddle the equilibrium.
            # Expand finitely toward the required signed command, within cap.
            direction = -1 if positive else 1
            step = max(self.config.torque_resolution_nm, abs(command_nm))
            self.candidate_nm = max(
                -self.cap_nm, min(self.cap_nm, command_nm + direction * step)
            )
            if abs(self.candidate_nm - command_nm) <= 1e-9:
                self.state, self.reason = "incomplete", "holding_authority_insufficient"
                return
        self.reason = (
            "verify_hold_again"
            if self._verification is not None
            else "next_bounded_probe"
        )
        self._check_trial_budget()

    def _check_trial_budget(self):
        if len(self.observations) + len(self.transitions) >= self.config.max_trials:
            self.state, self.reason = "incomplete", "holding_trial_budget_exhausted"

    def _propose_after_transition(self, command_nm, proposal, reason):
        proposal = max(-self.cap_nm, min(self.cap_nm, proposal))
        # A censored command supplies exploration history, never a signed
        # force bound. Subdivide a repeated proposal rather than cycle through
        # the same unobserved levels until the original trial budget expires.
        while any(abs(proposal-r["command_nm"]) <= 1e-9 for r in self.transitions):
            if abs(proposal-command_nm) <= 1e-9:
                self.state, self.reason = "incomplete", "holding_authority_insufficient"
                return
            proposal = (command_nm+proposal)/2
        self.candidate_nm, self.reason = proposal, reason
        self._verification = None
        self._interior_recheck_start = len(self.observations)
        self._check_trial_budget()

    def _probe_from_stable(self, stable, lower, upper):
        best = min(stable, key=abs)
        # Transition endpoints guide exploration only. They never enter the
        # resolved positive/negative motion bounds or stationary observations.
        motion_limit = upper if best < 0 else lower
        endpoints = [r["command_nm"] for r in self.transitions]
        if math.isfinite(motion_limit):
            endpoints.append(motion_limit)
        toward_zero = [
            max(best, min(0.0, v)) if best < 0 else min(best, max(0.0, v))
            for v in endpoints
            if (v - best) * best <= 0
        ]
        endpoint = (
            min(toward_zero, key=lambda v: abs(v - best)) if toward_zero else None
        )
        if abs(best) <= 1e-9 or (
            endpoint is not None
            and abs(endpoint - best) <= self.config.torque_resolution_nm
        ):
            self._verification = self.candidate_nm = best
            count = sum(
                r["outcome"] == "stationary" and abs(r["command_nm"] - best) <= 1e-9
                for r in self.observations
            )
            if count >= self.config.verification_repeats:
                self.state, self.reason = (
                    "complete",
                    "verified_usable_hold_not_unique_gravity",
                )
        else:
            self.candidate_nm = 0.0 if endpoint is None else (best + endpoint) / 2

    def record_transition(
        self, *, command_nm: float, requested_command_nm: float, evidence: dict
    ):
        """Refine a censored ramp without calling its endpoint a force bound."""
        if self.state != "searching":
            raise ValueError("holding search is terminal")
        if any(
            isinstance(v, bool) or not math.isfinite(v) or abs(v) > self.cap_nm + 1e-9
            for v in (command_nm, requested_command_nm)
        ):
            raise ValueError("holding transition must fit the command envelope")
        if abs(requested_command_nm - self.candidate_nm) > 1e-9:
            raise ValueError("holding transition must belong to the current probe")
        progress = evidence.get("net_displacement_rad")
        if self._recheck_interior_on_conflict and progress is not None and (
            isinstance(progress, bool) or not math.isfinite(progress)
        ):
            raise ValueError("holding transition progress must be finite")
        self.transitions.append(
            {
                **evidence,
                "command_nm": command_nm,
                "requested_command_nm": requested_command_nm,
                "verification_command_nm": self._verification,
                "quantity": "censored_transition_not_fixed_command_response",
            }
        )
        stable = self._eligible_stationary_commands()
        if not stable:
            if self._recheck_interior_on_conflict and self.observations:
                lower = max((r["command_nm"] for r in self.observations
                             if r["outcome"] == "negative_motion"), default=-math.inf)
                upper = min((r["command_nm"] for r in self.observations
                             if r["outcome"] == "positive_motion"), default=math.inf)
                if lower >= upper:
                    self.state, self.reason = "incomplete", "holding_response_not_repeatable_or_monotone"
                    return
                # A rejected recheck removes a holding claim, not the remaining
                # exploration interval. Reuse motion/transition proposals even
                # after a formerly stationary command has failed its return.
                # A censored endpoint is not the other side of a physical
                # bracket. If progress agrees with the only resolved motion
                # direction, explore away from it, including across zero.
                # This is a proposal only: the new command still needs a
                # constant-command observation at the original sample point.
                directed = (progress is not None
                            and abs(progress) >= self.config.movement_rad/2)
                one_sided_progress = directed and (
                    (math.isfinite(upper) and not math.isfinite(lower) and progress > 0)
                    or (math.isfinite(lower) and not math.isfinite(upper) and progress < 0)
                )
                if one_sided_progress:
                    bound = upper if progress > 0 else lower
                    step = max(self.config.torque_resolution_nm, abs(command_nm),
                               abs(command_nm-bound))
                    proposal = command_nm-math.copysign(step, progress)
                    self._propose_after_transition(command_nm, proposal,
                        "explore_unverified_hold_beyond_censored_endpoint")
                    return
                # Opposing or unresolved transient progress can propose an
                # interior test, but never certify a bracket's resolution.
                left = right = command_nm
                if math.isfinite(lower) and math.isfinite(upper):
                    left, right = lower, upper
                elif math.isfinite(lower) and command_nm > lower:
                    left, right = lower, command_nm
                elif math.isfinite(upper) and command_nm < upper:
                    left, right = command_nm, upper
                if math.isfinite(left) and math.isfinite(right) and right > left:
                    self._propose_after_transition(command_nm, (left+right)/2,
                        "refine_unverified_hold_between_motion_and_transition")
                    return
            if (self._recheck_interior_on_conflict and not self.observations
                and self._verification is None and progress is not None
                and abs(progress) >= self.config.movement_rad/2):
                # An unverified arrival command is not yet a holding anchor.
                # Censored progress guides a new bounded proposal only; it
                # never enters the resolved force bounds or static samples.
                proposal = (command_nm/2 if progress*command_nm > 0 else
                            command_nm-math.copysign(max(self.config.torque_resolution_nm,
                                                       abs(command_nm)), progress))
                self._propose_after_transition(command_nm, proposal, "refine_unverified_arrival_command")
                return
            self.state, self.reason = (
                "incomplete",
                "holding_no_repeatable_anchor" if self._verification is not None
                else "holding_transition_without_stable_anchor",
            )
            return
        best = min(stable, key=abs)
        # A failed return check is not a signed motion bound. Keep its old
        # observation, but verify another known hold instead of replaying this
        # same transition until the attempt budget expires.
        # With no resolved withdrawal step, also recheck before exploring.
        if self._verification is not None or (command_nm - best) * best >= 0:
            self._verification = self.candidate_nm = best
        else:
            positive = [
                r["command_nm"]
                for r in self.observations
                if r["outcome"] == "positive_motion"
            ]
            negative = [
                r["command_nm"]
                for r in self.observations
                if r["outcome"] == "negative_motion"
            ]
            self._probe_from_stable(
                stable,
                max(negative, default=-math.inf),
                min(positive, default=math.inf),
            )
        # A censored transition is not a successful final trial. Return to the
        # point and confirm the selected hold again before declaring completion.
        if self.state == "complete":
            self.state = "searching"
        if self._interior_recheck_start is not None:
            self._interior_recheck_start = len(self.observations)
        self.reason = (
            "verify_hold_after_transition"
            if self._verification is not None
            else "refine_censored_transition"
        )
        self._check_trial_budget()

    def _eligible_stationary_commands(self):
        rejected = [r["verification_command_nm"] for r in self.transitions
                    if r.get("verification_command_nm") is not None]
        lower = max((r["command_nm"] for r in self.observations
                     if r["outcome"] == "negative_motion"), default=-math.inf)
        upper = min((r["command_nm"] for r in self.observations
                     if r["outcome"] == "positive_motion"), default=math.inf)
        return [r["command_nm"] for r in self.observations
                if r["outcome"] == "stationary"
                and (not self._recheck_interior_on_conflict or lower < r["command_nm"] < upper)
                and not any(abs(r["command_nm"] - v) <= 1e-9 for v in rejected)]

    @property
    def return_command_nm(self):
        """Use the selected recheck hold during reacquisition, not a rejected one."""
        if self._verification is not None:
            return self._verification
        return min(self._eligible_stationary_commands(), key=abs,
                   default=None)

    def status(self, *, detailed=False):
        stable = [
            r["command_nm"] for r in self.observations if r["outcome"] == "stationary"
        ]
        view = {
            "state": self.state,
            "reason": self.reason,
            "candidate_nm": self.candidate_nm,
            "trial_count": len(self.observations),
            "transition_count": len(self.transitions),
            "max_trials": self.config.max_trials,
            "movement_rad": self.config.movement_rad,
            "return_command_nm": self.return_command_nm,
            "usable_hold_nm": self._verification if self.state == "complete" else None,
            "observed_stationary_range_nm": [min(stable), max(stable)]
            if stable
            else None,
            "negative_motion_below_nm": max(
                (
                    r["command_nm"]
                    for r in self.observations
                    if r["outcome"] == "negative_motion"
                ),
                default=None,
            ),
            "positive_motion_above_nm": min(
                (
                    r["command_nm"]
                    for r in self.observations
                    if r["outcome"] == "positive_motion"
                ),
                default=None,
            ),
            "quantity": "software_command_not_measured_force",
            "auto_apply": False,
            "identifies_unique_gravity": False,
            "verification_semantics": "same_pose_command_rechecks_not_independent_history_validation",
            "interior_recheck_enabled": self._recheck_interior_on_conflict,
            "interior_recheck_start_observation": self._interior_recheck_start,
            "rejected_stationary_commands_nm": sorted(set(stable)-set(self._eligible_stationary_commands())),
        }
        if detailed:
            view["observations"] = [dict(r) for r in self.observations]
            view["transitions"] = [dict(r) for r in self.transitions]
        return view
