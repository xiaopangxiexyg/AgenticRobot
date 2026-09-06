# Agent Gateway Instructions

`agent_gateway` is the upper-framework-neutral attachment and Skill execution
control plane. Read `agent_gateway/README.md` and use
`$change-agent-gateway` for protocol, runtime, transport, Skill, Route, Artifact,
or concrete-profile changes.

## Ownership

- `protocol/`, `runtime/`, and `transport/` own versioned wire contracts,
  admission and reconciliation, bounded operation state, authenticated
  transport, and exact executor registration. They must not import upper
  planners or concrete control, simulation, hardware, teleoperation, or model
  deployments.
- `skills/` owns target-independent semantic definitions and strict I/O
  schemas. It must not select devices, frames, safety envelopes, deployments,
  checkpoints, or controller parameters.
- `profiles/` is the concrete application edge. Profiles may bind application
  resources and lower executors through typed ports, but must not copy lower
  safety, IK, reference, dispatch, HOLD, or episode owners into the Gateway.
- The Gateway owns no RPent Task, Program, Memory, Experience, canonical robot
  authority, safety, IK, reference, dispatch, or durable dataset lifecycle.

Import direction, public protocol identity, schema hashes, Route registration,
and compatibility shims are enforced by code and focused tests. Current
backends, feature availability, file inventories, calibration values, and
migration status belong in profile documentation or versioned manifests, not
this file.

## Effect and authority

- Inspection and perception Routes remain strictly read-only and cannot call a
  motion, lifecycle, command-admission, dispatch, or domain-HOLD port.
- Real motion is default-off. An active request requires the exact admitted
  profile and Route, explicit operator-derived authorization, a fresh
  request-bound grant, fresh lower state, and lower-owned execution.
- Admission, cancellation ACK, transport return, disconnect, timeout, or an
  operation terminal state is never lower-confirmed HOLD. Preserve the exact
  lower receipt and reconciliation state.
- Frame identity and conversion have one owner. Profiles bind explicit frame
  graphs; Gateway and Agent code must not duplicate tool geometry or reinterpret
  teleoperation calibration as Station authority.
- Application resources such as devices, models, stores, and workers start
  before the transport is published and close under the application lifecycle.
  A Skill invocation must not create, reset, restart, or close them.

## Skill and Artifact contracts

Use the protocol registration template and strict schemas so every advertised
Skill, Route, I/O contract, resource claim, and deployment binding closes over
the same hashes. Artifact identity is immutable metadata plus a content hash;
paths and transport grants are short-lived access mechanisms, not durable
identity. The Gateway store is a transient handoff cache, never the dataset,
Experience, or Memory store.

Upgrade the single public v2 contract in place. Do not add a parallel legacy
transport or consumer mode. Keep historic paths as stateless forwarding shims
only when compatibility is still required.

## Verification

Run focused Gateway contract tests through `bash scripts/run_pytest.sh ...`.
Validate the neutral core separately from every changed profile. Contract or
in-memory tests prove only the boundary exercised; they do not prove deployed
cell geometry, production launcher ownership, simulator readiness, or safe real
motion.
