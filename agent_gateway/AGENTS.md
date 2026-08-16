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
  `profiles/motion_primitive_resolution.py` is the shared pure activation-time
  seam for applying a reviewed primitive to lower-measured state; target-
  specific frame/profile identities remain in the concrete profile. MuJoCo's
  reviewed 6+3 primitive identities and parameter profiles live in
  `profiles/mujoco/motion_primitive_profiles.py`, separate from Route-manifest
  construction and executor lifecycle. The real 6+3 application follows the
  same split under `profiles/real/motion_primitive_*`; its factory is
  attach-only and must share exactly one lower `RealAgentMotionPort` with the
  returned RealTeleop component aggregate.

Semantic primitive axes and lower execution coordinates must remain explicit.
When a Skill is defined in canonical Station/desk axes but the lower owner
executes in an arm-base frame, the concrete profile must bind an exact typed
frame graph and transform the measured anchor and resolved target across that
graph. Do not attach a canonical axis contract to a rotated arm-base frame.
Each motion Route must pin exactly one compatible lower safety envelope; a
larger fixed-orientation range must not silently widen the generic/cardinal
envelope.

Real motion remains default-off. The exact profile must require a one-use,
request-bound operator Grant, a fresh lower-measured pose cut, a RUNNING/READY
lower owner cut, and terminal lower-confirmed HOLD. Do not reinterpret the
existing foot switches as an Agent deadman lease. The lower RealTeleop control
thread alone may apply activation/HOLD requests, construct commands through the
existing coupled target solver, publish already-read measured qpos/FK cuts, and
pass commands through the existing filter, guard, admission, rig dispatch, and
HOLD boundaries. A caller must continue with the component aggregate returned
by the attach factory; the original aggregate is not attached.

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
and do not infer simulator or real-hardware readiness from contract tests. An
in-memory lower-owner loop proves wiring and fail-HOLD semantics only; it is not
evidence that the production launcher supplies authoritative owner identities,
that a safety envelope matches the deployed cell, or that any real motion was
executed safely.
