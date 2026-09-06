---
name: remote-access
description: Operate or diagnose AgenticRobot remote hosts, remote execution, devices, or hardware through the codex_remote wrappers. Use for remote identity, environment, sync, tests, logs, camera reads, device checks, process control, result retrieval, safety preflight, or any requested remote robot action; keep local-only work out of this workflow.
---

# Remote Access

Use the reviewed `codex_remote/` wrappers as the execution boundary. Read
[the wrapper catalog](references/wrapper-catalog.md) when selecting a wrapper
or classifying an operation. Read
[the remote design](../../../codex_remote/README.md) before modifying wrappers
or shared safety behavior.

## Workflow

1. **Fix the context.** Record the exact SSH target, repository path or runtime
   surface, intended effect, local/remote write scope, and timeout or stop
   condition. Distinguish the SSH host from a robot/controller address.
2. **Classify the effect.** Separate read-only observation, result transfer,
   process execution, device-state change, and real motion. Treat an unknown or
   mixed effect as control-risk until a reviewed wrapper classifies it.
3. **Verify identity.** Pass `-H/--host` explicitly. For a non-default target,
   run the identity wrapper before process, device, or hardware work. Never
   fall back to another target after failure.
4. **Use the narrow wrapper.** Prefer the wrapper dedicated to the requested
   action. Do not handwrite SSH for remote writes, process control, hardware
   access, USB/host reset, or robot control.
5. **Apply the gate.**
   - Run relevant read-only diagnostics directly.
   - State target and write scope before non-motion transfer.
   - State process and timeout before remote execution.
   - Require explicit current-request authority before device-state changes or
     motion.
   - Run a fresh safety preflight and pass its bounded token only to the exact
     authorized control-risk command.
6. **Keep code ownership local.** When confirmed no-delete realtime sync is
   active, edit locally and use the remote only for the environment, devices,
   drivers, GPU, or execution evidence it owns. Verify sync when freshness is
   material.
7. **Collect bounded evidence.** Remote output does not imply local
   materialization. Inspect or retrieve named results intentionally and report
   the exact target, command class, boundary reached, and remaining unverified
   state.

## Guardrails

- Do not add `--allow-control-risk` merely to make a generic command run.
- Do not infer safe realized motion from process exit, SDK acceptance, timeout,
  disconnect, or a transport response.
- Do not reuse a prior target's path, environment, device conclusions, safety
  token, or write scope after the target changes.
- Do not store credentials or host secrets in the repository.
- Modify the wrappers or their policy only when that surface is explicitly
  requested; preserve fail-closed defaults and run focused syntax/tests.
