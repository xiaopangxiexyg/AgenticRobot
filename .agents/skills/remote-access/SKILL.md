---
name: remote-access
description: Operate, diagnose, or change AgenticRobot remote-access workflows through the codex_remote wrappers. Use when a task crosses the local/remote boundary for identity, environment, sync, tests, logs, camera capture, device checks, process execution, result retrieval, wrapper maintenance, or an explicitly requested remote robot action; do not use for local-only code, motion-path logic, Gateway contracts, or local-resource layout.
---

# Remote Access

Own only the operation that crosses the remote boundary. Local code semantics
remain with their domain Skill; `$debug-teleop-ik-motion` owns motion-path logic,
`$change-agent-gateway` owns Gateway contracts, and
`$manage-project-local-resources` owns local storage and bindings.

Use the reviewed `codex_remote/` wrappers as the execution boundary. For a known
wrapper, inspect its `--help`; read
[the wrapper catalog](references/wrapper-catalog.md) when the choice or effect
class is unclear. Before changing a wrapper, inspect that wrapper, its shared
helper and tests, and only the relevant section of
[the remote design](../../../codex_remote/README.md).

## Workflow

1. **Fix the context.** Record the exact SSH target, repository path or runtime
   surface, intended effect, local/remote write scope, and timeout or stop
   condition. Distinguish the SSH host from a robot/controller address.
2. **Select the narrow wrapper.** Use the known dedicated wrapper or consult the
   catalog. Separate read-only observation, result transfer, process execution,
   device-state change, and real motion.
3. **Verify identity.** Pass `-H/--host` explicitly. For a non-default target,
   run the identity wrapper before process, device, or hardware work. Never
   fall back to another target after failure.
4. **Apply the operation contract.**
   - Run relevant read-only diagnostics through the corresponding wrapper.
   - State target and write scope before non-motion transfer.
   - State process and timeout before remote execution.
   - Require explicit current-request authority before device-state changes or
     motion.
   - Run a fresh safety preflight and pass its bounded token only to the exact
     authorized control-risk command.
5. **Keep code ownership local.** When confirmed no-delete realtime sync is
   active, edit locally and use the remote only for the environment, devices,
   drivers, GPU, or execution evidence it owns. Verify sync when freshness is
   material.
6. **Collect bounded evidence.** Remote output does not imply local
   materialization. Inspect or retrieve named results intentionally and report
   the exact target, command class, boundary reached, and remaining unverified
   state.

## Guardrails

- Do not add `--allow-control-risk` merely to make a generic command run.
- Do not infer safe realized motion from process exit, SDK acceptance, timeout,
  disconnect, or a transport response.
- Do not reuse a prior target's path, environment, device conclusions, safety
  token, or write scope after the target changes.
- Keep host-specific configuration and credentials outside the repository;
  copying them between trusted development machines is a separate, explicit
  local-resource or provisioning operation.
- Modify the wrappers or their policy only when that surface is explicitly
  requested; preserve fail-closed defaults and run focused syntax/tests.
