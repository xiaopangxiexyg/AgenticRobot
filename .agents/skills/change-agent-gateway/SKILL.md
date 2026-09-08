---
name: change-agent-gateway
description: Change or review AgenticRobot Gateway protocol, admission, transport, semantic Skills, Routes, Artifact I/O, compatibility, or concrete profile bindings. Use when a Gateway-owned contract changes, not for unrelated motion or remote execution.
---

# Change Agent Gateway

Use this Skill only to route a Gateway-owned change. `agent_gateway/AGENTS.md`
is the single authority for owner boundaries, effects, artifacts, admission,
safety, and receipts; read it before editing. Locate only the relevant
[Gateway README](../../../agent_gateway/README.md) headings and current symbols.
Use `$remote-access`, `$debug-teleop-ik-motion`, or
`$manage-project-local-resources` only when the task actually crosses those
separate boundaries.

## Workflow

1. Identify the real producer, consumer, and primary owner from the directory
   map in `agent_gateway/AGENTS.md`.
2. Trace the current definition through registration, runtime, executor, and
   caller; inspect the focused tests and generated catalog when affected.
3. Change the owning boundary first, then update current consumers and the
   versioned manifest, registry, or profile documentation that exposes it.
4. Preserve construction/admission failure for invalid combinations and the
   scoped safety and receipt contracts.

## Validation

Run focused owner and consumer tests through `bash scripts/run_pytest.sh ...`;
include the applicable rejection, timeout, or cancellation path. Report the
exact boundary tested. Contract or in-memory evidence does not establish
deployment, simulator, or hardware readiness.
