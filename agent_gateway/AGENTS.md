# Agent Gateway Working Agreement

This package is the DexTeleop-owned, upper-framework-neutral attachment and
Skill execution control plane. It is not part of the Control Station GUI and
must remain usable without importing `control_station` unless a concrete
profile explicitly binds a Station application port.

## Ownership

- `protocol/` owns versioned wire contracts, Skill/schema catalogs, Route and
  deployment references, and transport-neutral serialization validation.
- `runtime/` owns authenticated-client admission, idempotency/reconciliation,
  bounded operation projection, exact Route-to-executor registration, and
  cancellation/HOLD requests through injected lower ports.
- `transport/` authenticates and bounds connections, then delegates to one
  bound runtime facade. It does not reinterpret Skill or motion semantics.
- `skills/` owns target-independent Skill definitions and payload/result
  schemas. It must not select concrete devices, frames, safety envelopes,
  routes, deployments, or controller parameters. Pure motion primitives remain
  in `robot_motion.primitives`; Skills may name bounded semantic choices but
  must not duplicate primitive execution, safety, IK, or dispatch.
- `profiles/` owns concrete application and deployment bindings. A profile may
  import `control_station`, `sim`, deployment adapters, and lower motion APIs
  only to assemble one exact application leaf through typed ports.

Semantic primitive axes and lower execution coordinates must remain explicit.
When a Skill is defined in canonical Station/desk axes but the lower owner
executes in an arm-base frame, the concrete profile must bind an exact typed
frame graph and transform the measured anchor and resolved target across that
graph. Do not attach a canonical axis contract to a rotated arm-base frame.
Each motion Route must pin exactly one compatible lower safety envelope; a
larger fixed-orientation range must not silently widen the generic/cardinal
envelope.

The Gateway never owns an upper planner, RPent Task/Program/Memory/Experience,
canonical robot authority, safety, IK, reference, dispatch, or dataset episode
lifecycle. Admission, an operation terminal state, a disconnect ACK, or a
transport response must not be presented as lower-confirmed HOLD evidence.

## Compatibility and verification

Historic `control_station.agent_gateway`,
`control_station.sim_hil_profile.agent`, and Station CubeLift Skill paths are
forwarding shims only. The historic RealTeleop observation Gateway module is
also a compatibility re-export. The former
`agent_gateway.protocol.motion_schemas` owner is a compatibility re-export for
`agent_gateway.skills.motion.schemas`. New production imports and tests must
use the owning `agent_gateway.*` path; shims must not duplicate implementation,
state, schemas, or resource ownership.

Preserve public wire field names, schema/hash domains, Skill/Route/deployment
identities, idempotency behavior, and fail-closed cancellation semantics during
refactors. Validate the neutral core separately from each concrete profile,
and do not infer simulator or real-hardware readiness from contract tests.
