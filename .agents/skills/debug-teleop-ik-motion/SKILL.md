---
name: debug-teleop-ik-motion
description: Diagnose or change AgenticRobot pose-to-command motion, IK, reference, interpolation, dispatch, or stateful modifiers; compare offline, shadow, and Active evidence. Use for motion-path faults, not camera discovery or Gateway wire contracts.
---

# Debug Teleoperation IK Motion

Own the motion path from source pose through accepted reference, dispatch, and
measured feedback. Use `$remote-access` only for a required remote/device
operation, `$change-agent-gateway` only when a Gateway-owned contract changes,
and `$manage-project-local-resources` only when evidence placement or migration
is itself part of the task.

Locate the first authoritative divergence before changing downstream behavior.
Read [motion contracts](references/motion-contracts.md) for frame, time, IK, and
stateful-modifier rules. Read
[evidence and metrics](references/evidence-and-metrics.md) when comparing traces
or validating a candidate.

## Select the mode

- **Diagnose:** inspect evidence and identify the first supported causal
  boundary; do not edit.
- **Review:** evaluate a proposal or diff for contract violations and missing
  cases; do not silently implement it.
- **Implement:** make the smallest structurally complete, evidence-supported
  change. Use a known Git revision or deployment rollback for recovery; add a
  selectable runtime fallback only for an identified operational requirement.
- **Validate:** compare a named baseline and candidate under equivalent inputs
  and declared acceptance criteria.

Do not escalate from diagnosis or review to implementation without a requested
change.

## Workflow

1. **Fix provenance.** Identify the arm, runtime/profile, symptom interval,
   current implementation, desired contract, and at least one inspectable log,
   trace, trajectory, diff, or code path. With insufficient evidence, bound the
   likely stage and request the smallest discriminating artifact.
2. **Map authority.** Trace:

   ```text
   operator/source pose
     -> shaping and frame conversion
     -> base target
     -> stateful modifiers and composed target
     -> IK candidates
     -> selection and safety admission
     -> accepted reference
     -> interpolation/trajectory
     -> dispatch
     -> measured feedback
   ```

   Label observations, proposals, accepted references, commands, and realized
   motion. Reject any observer, cache, debug publisher, or replay adapter acting
   as a second state owner.
3. **Align contracts.** Verify frame identities, timestamps, sample ages,
   quaternion continuity, update clocks, lower profile, and the point at which
   each state becomes authoritative. Do not compare differently framed or
   differently timed values as if aligned.
4. **Find the first divergence.** Walk from source to measured feedback and
   quantify the earliest abnormal transition. Do not compensate in a later
   stage until the upstream signal is demonstrated sound.
5. **Classify the mechanism.** Separate transformation, IK generation,
   selection/admission, reference evolution/rate conversion, and
   dispatch/execution faults. Treat singularity, branch competition, joint
   limits, stale state, residual accumulation, and deadline misses as hypotheses
   until current evidence distinguishes them.
6. **Inspect stateful terms.** When modifiers are present, write the actual
   composition equation and record every persistent/transient term, owner,
   frame, update clock, reset/HOLD/stale behavior, and bound. Validate the final
   sum, not only individual components.
7. **Change reversibly.** Keep candidate generation, selection, smoothing, and
   admission separable. Preserve the baseline revision and its evidence.
   Add bounded reason codes or debug fields instead of unbounded hot-path logs.
   Never weaken or incidentally change canonical collision, force-stop, or
   emergency-stop behavior to make a motion-path candidate pass.
8. **Validate the hypothesis.** Run the narrowest test or replay that can
   falsify the proposed cause, then broaden only across affected boundaries.
   Compare hard validity, pose error, branch behavior, dynamics, deadlines, and
   representative normal motion.

## Execution boundary

- This Skill may inspect or change code and run offline tests, replay, and
  simulation. Remote processes, devices, and hardware belong to
  `$remote-access` and the scope of the current request.
- After unexpected real motion, freeze the implicated Active path, preserve the
  exact commands, configuration, logs, processes, and traces, and return to
  read-only diagnosis.
- Treat ambiguous frames, NaN, invalid FK, impossible time, or corrupt evidence
  as a failed diagnostic input rather than tuning around it.

## Report

State the selected mode, strongest confirmed boundary, decisive evidence,
remaining alternatives, confidence, and next safe action. For implementation,
also report the recovery path and exact validation run. Distinguish offline, replay,
simulation, shadow, and real-hardware evidence.
