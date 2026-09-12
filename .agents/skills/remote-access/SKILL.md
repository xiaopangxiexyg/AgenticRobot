---
name: remote-access
description: Use AgenticRobot codex_remote wrappers for remote inspection, transfer, execution, device checks, or wrapper maintenance. Applies when crossing the local/remote boundary; local domain logic remains with its owner.
---

# Remote Access

Own the remote operation; local semantics remain with the domain Skills named
in the repository's `AGENTS.md`.

Use reviewed `codex_remote/` wrappers. Inspect `--help` for unknown, changed, or
rejected arguments; reuse checked arguments for the same version. Read
[the catalog](references/wrapper-catalog.md) only when wrapper choice or effect
class is unclear. For maintenance, apply `codex_remote/AGENTS.md` and inspect
the affected wrapper, helper, tests and relevant
[design section](../../../codex_remote/README.md).

## Workflow

1. State the exact SSH target, project/runtime surface, intended effect,
   local/remote write scope and timeout/stop condition once per bounded batch.
   Distinguish SSH hosts from robot/controller addresses. Subsequent commentary
   may reference this binding and report changes. Re-expand unclear or changed
   bindings and requests for device/motion authority; prose reuse grants no
   additional authority.
2. Pass `-H/--host` explicitly on every call and preserve wrapper audit output.
   For non-default targets, run the identity wrapper before process, device or
   hardware work. A printed `[target]` is not identity verification. Never fall
   back to another target after failure.
3. Select the narrow wrapper for observation, transfer, process execution,
   device changes or motion. Before transfer, state target and write scope;
   before execution, state process and timeout (the batch statement may suffice).
   Device changes and motion require explicit current-request authority.
   Control-risk commands require fresh safety preflight and its bounded token
   for the exact authorized command.
4. With confirmed no-delete realtime sync, edit locally, verify sync when
   freshness matters, and use the remote for its environment/devices/execution.
   Inspect or retrieve named outputs deliberately; they do not imply local
   materialization.
5. Report results, errors, exit status and unverified state without repeating
   unchanged connection metadata or full argv. Final handoffs must independently
   identify the exact target, command class and boundary reached.

For cwd/Python checks use `remote_env.sh --summary`; omit it when PATH, conda
inventory, GPU or CUDA diagnostics are needed. Unchecked fields are not healthy
by implication. Avoid unrelated inventories on each SSH call; required identity,
sync and safety checks still apply. Bound logs with supported `--head`, `--tail`
or `--max-bytes`, disclose partial evidence, and never infer success from omissions.
For text inspection use `remote_view.sh --compact`; omit it when full file metadata
is needed. Its omitted-byte count covers selected lines, not the whole file.

## Guardrails

- Do not add `--allow-control-risk` merely to make a generic command run.
- Do not infer safe realized motion from process exit, SDK acceptance, timeout,
  disconnect, or a transport response.
- Do not reuse a prior target's path, environment, device conclusions, safety
  token, or write scope after the target changes.
- Treat an unknown SSH host key as an onboarding boundary. Do not weaken normal
  wrapper host-key checking to make a connection succeed.
- Keep host-specific configuration and credentials outside the repository;
  copying them between trusted development machines is a separate, explicit
  local-resource or provisioning operation.
- Preserve fail-closed defaults; changes to allowed effects or safety policy
  require an explicit decision.
