---
name: debug-teleop-ik-motion
description: Diagnose or change the AgenticRobot pose-to-command motion path, including teleoperation jitter, sticking, catch-up, wrong-range or continued motion, IK branch and singularity faults, joint limits, interpolation, stateful modifiers, Active-versus-shadow gaps, and replay/deployment Pose-to-QPOS differences. Do not use for primary camera, device discovery, GUI, dataset schema, collision-model authoring, or hardware operation tasks.
---

# Debug Teleoperation IK Motion

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
- **Implement:** make the smallest reversible, evidence-supported change and
  preserve a selectable fallback.
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
   admission separable. Preserve the stable path unless promotion is requested.
   Add bounded reason codes or debug fields instead of unbounded hot-path logs.
   Never weaken or incidentally change canonical collision, force-stop, or
   emergency-stop behavior to make a motion-path candidate pass.
8. **Validate the hypothesis.** Run the narrowest test or replay that can
   falsify the proposed cause, then broaden only across affected boundaries.
   Compare hard validity, pose error, branch behavior, dynamics, deadlines, and
   representative normal motion.

## Incident boundary

- Do not start a device or command motion through this diagnostic Skill. Use
  `$remote-access` for separately authorized remote/hardware operation.
- After unexpected real motion, freeze the implicated Active path, preserve the
  exact commands, configuration, logs, processes, and traces, and return to
  read-only diagnosis.
- Stop on ambiguous frame/profile/authority, NaN, invalid FK, impossible time,
  corrupt evidence, or failed hard certificates.
- Do not tune and retry while the first divergence or missing safety invariant
  remains open.

## Report

State the selected mode, strongest confirmed boundary, decisive evidence,
remaining alternatives, confidence, and next safe action. For implementation,
also report the fallback and exact validation run. Distinguish offline, replay,
simulation, shadow, and real-hardware evidence.
