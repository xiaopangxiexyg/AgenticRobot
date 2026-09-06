---
name: change-agent-gateway
description: Change or review AgenticRobot agent_gateway protocol, runtime, transport, semantic Skills, Routes, Artifact I/O, compatibility shims, or concrete simulation and real profiles. Use when adding or modifying a Gateway capability, schema, registration, executor binding, resource lifecycle, frame binding, admission, cancellation, HOLD, or protocol compatibility behavior.
---

# Change Agent Gateway

Preserve one neutral Gateway core and one lower authority owner. Read
[the Gateway README](../../../agent_gateway/README.md) and the nearest profile
documentation for the changed surface.

## Classify the change

Choose one primary owner before editing:

- `protocol/`: versioned wire types, schemas, hashes, catalogs, and
  transport-neutral validation.
- `runtime/`: admission, idempotency, reconciliation, operations, workers,
  access policy, cancellation, and lower-port requests.
- `transport/`: authenticated bounded transport over one runtime facade.
- `skills/`: target-independent semantic definitions and strict I/O schemas.
- `profiles/`: concrete application resources, Route/deployment binding,
  executor construction, frames, observation, evaluation, and lifecycle.
- compatibility path: stateless forwarding only; never a second schema, state,
  worker, service, or resource owner.

If the change spans owners, define the public contract and dependency order
before implementing downstream bindings.

## Workflow

1. Trace the existing definition, schema, Route, registration, runtime facade,
   executor, resource owner, and caller. Inspect current tests and generated
   catalogs rather than relying on prose feature lists.
2. Separate semantic Skill choice from trusted Route, deployment, frame,
   resource, safety, and controller binding.
3. Modify the owning type, schema, registration template, or profile first.
   Make invalid combinations fail closed at construction or admission.
4. Preserve exact version/hash domains, idempotency, cancellation, timeout,
   reconciliation, and lower-receipt semantics. Upgrade the public v2 surface
   in place unless an explicitly reviewed version transition is requested.
5. Keep inspection/perception strictly read-only. Keep real motion default-off
   and require exact operator-derived authorization, request-bound grant, fresh
   lower state, lower owner readiness, and lower-confirmed terminal HOLD.
6. Keep application resources outside Skill invocation. Start/admit them before
   publishing transport; close them through one reversible application
   lifecycle.
7. Update current backend or capability information in versioned manifests,
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
