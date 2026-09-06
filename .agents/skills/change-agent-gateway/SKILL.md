---
name: change-agent-gateway
description: Change or review a proposed change to AgenticRobot agent_gateway-owned protocol, runtime, transport, semantic Skill, Route, Artifact I/O, compatibility, or concrete profile bindings. Use when a Gateway capability, schema, registration, executor binding, resource lifecycle, frame binding, admission, cancellation, HOLD, or public compatibility contract changes; do not use merely because a task runs remotely or debugs motion outside a Gateway boundary.
---

# Change Agent Gateway

Own changes to the Gateway contract and its application bindings. Use
`$remote-access` for remote execution, `$debug-teleop-ik-motion` for the
pose-to-command motion path, and `$manage-project-local-resources` for ignored
resource placement; combine Skills only when the task crosses those boundaries.

Treat `agent_gateway/AGENTS.md` as the invariant contract. Locate relevant
[Gateway README](../../../agent_gateway/README.md) sections by heading or symbol
and inspect the current code, registration, profile documentation, and tests for
the changed surface. Do not load the full README by default.

## Classify the change

Choose one primary owner before editing:

- `protocol/`: wire types, schemas, hashes, and catalogs.
- `runtime/`: admission, operation state, reconciliation, and lower-port calls.
- `transport/`: authenticated transport over the runtime facade.
- `skills/`: target-independent semantics and strict I/O schemas.
- `profiles/`: concrete resources, Routes, executors, frames, and lifecycle.
- compatibility path: stateless forwarding to the current public contract.

If the change spans owners, define the public contract and dependency order
before implementing downstream bindings.

## Workflow

1. Trace the existing definition, schema, Route, registration, runtime facade,
   executor, resource owner, and caller. Inspect current tests and generated
   catalogs rather than relying on prose feature lists.
2. Separate semantic Skill choice from trusted Route, deployment, frame,
   resource, safety, and controller binding.
3. Modify the owning type, schema, registration template, or profile first.
4. Preserve the scoped AGENTS contracts for protocol identity, admission,
   effects, lower receipts, resource lifecycle, and compatibility. Make invalid
   combinations fail at construction or admission rather than in downstream
   execution.
5. Update current backend or capability information in versioned manifests,
   registries, or profile documentation, not in `AGENTS.md`.

## Validation

Run tests through `bash scripts/run_pytest.sh ...`.

- Protocol/schema: registration, catalog, hash/snapshot, serialization, and
  compatibility tests.
- Runtime/transport: admission, reconciliation, operation store, events,
  authentication, endpoint, and disconnect/timeout tests.
- Artifact I/O: slot, store, transfer, quota, hash/length, grant, receipt, gap,
  and collection tests.
- Concrete profile: neutral import boundary plus that profile's lifecycle,
  frame, resource, executor, and failure tests.
- Real motion: authorization, single-use grant, fresh state, safety envelope,
  control-owner, fault, timeout, disconnect, and confirmed-HOLD tests.

Report the exact boundary tested. In-memory or contract validation does not
establish simulator, deployment, or real-hardware readiness.
