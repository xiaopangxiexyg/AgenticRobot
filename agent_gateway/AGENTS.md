# Agent Gateway Working Agreement

This package is the DexTeleop-owned, upper-framework-neutral attachment and
Skill execution control plane. It is not part of the Control Station GUI and
must remain usable without importing `control_station` unless a concrete
profile explicitly binds a Station application port.

## Ownership

- `protocol/` owns versioned wire contracts, Skill/schema catalogs, Route and
  deployment references, generic Skill I/O/Artifact/Event contracts, the
  reviewed Skill registration template, and transport-neutral serialization
  validation. The only current public Skill protocol is
  `dexteleop.agent_skill.v2`; do not create parallel v1/v2 Skill families.
- `runtime/` owns authenticated-client admission, idempotency/reconciliation,
  bounded operation projection, exact Route-to-executor registration, and
  cancellation/HOLD requests through injected lower ports. Its session-level
  protocol runtime binds one manifest, I/O catalog, capability document,
  replay journal, and optional transient Artifact store/data plane.
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
  assembled with exactly one lower `RealAgentMotionPort`. The preferred real
  Skill deployment factory shares that port with an independent exclusive
  lower owner under `teleop_runtime.skill_deploy`; it must not construct or
  import RealTeleop, VIVE, collection, GUI, or `control_composition`. The
  historic RealTeleop attachment remains a compatibility path, not the owner
  of new real Skill deployment architecture.

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
existing foot switches as an Agent deadman lease. In the preferred deployment,
the independent Skill control owner alone may apply activation/HOLD requests,
construct commands through the lower coupled target solver, publish already-read
measured qpos/FK cuts, and pass commands through the lower filter, guard,
admission, dispatch, and HOLD boundaries. It must reuse those lower owners
through typed application ports rather than copying them into the Gateway. On
the compatibility attachment only, the existing RealTeleop control thread
retains those effects and the caller must use the replacement component
aggregate returned by the attach factory.

The Gateway never owns an upper planner, RPent Task/Program/Memory/Experience,
canonical robot authority, safety, IK, reference, dispatch, or dataset episode
lifecycle. Admission, an operation terminal state, a disconnect ACK, or a
transport response must not be presented as lower-confirmed HOLD evidence.

## Generic Skill I/O and Artifact lifecycle

Every advertised Skill has exactly one `SkillIOContract`, including Skills
whose input and output Artifact slot lists are empty. New Skills should be
assembled with `protocol.skill_registration.SkillRegistrationTemplate` so the
Definition, strict schemas, I/O contract, and Routes fail closed over the same
hashes. Hardware-information Skills use ordinary typed Artifact slots; do not
add device- or depth-specific transport methods.

Artifact identity is a finite immutable descriptor and content hash. A path,
URL, file descriptor, or object-store key may appear only in a short-lived,
authenticated access grant. `local_path` is valid only for a reviewed
same-host trust boundary and is never a durable identity. The Gateway store is
a recoverable transient handoff cache: stage and reserve quota, transfer,
verify hash/length, atomically commit, bind to an operation, serve through an
ephemeral grant, accept an upper-owned materialization receipt, then collect
after pins/retention expire. It does not become the dataset, Experience, or
Memory store.

The current data plane implements only `chunked_rpc` and same-host
`local_path`. Uploads always use `chunked_rpc`; `local_path` is a read-only
grant over committed bytes. Do not advertise speculative transfer mechanisms
or unenforced slot security/retention metadata. Multi-file results use multiple
bindings or one explicitly encoded archive until a concrete consumer justifies
another contract.

Artifact transfer is capability-negotiated within v2. Inline-only sessions
publish `artifact_transfer_enabled=false`; a profile that advertises Artifact
slots must inject a matching store, broker, limits, and at least the bounded
`chunked_rpc` fallback. Events are bounded at-least-once hints with monotonic
session sequence numbers and explicit replay gaps. Operation query/catalog
discovery remains authoritative after a gap; event payloads must never carry
large Artifact bodies.

## Compatibility and verification

Historic `control_station.agent_gateway`,
`control_station.sim_hil_profile.agent`, and Station CubeLift Skill paths are
forwarding shims only. The historic RealTeleop observation Gateway module is
also a compatibility re-export. The former
`agent_gateway.protocol.motion_schemas` owner is a compatibility re-export for
`agent_gateway.skills.motion.schemas`. New production imports and tests must
use the owning `agent_gateway.*` path; shims must not duplicate implementation,
state, schemas, or resource ownership.

Preserve the single v2 public wire field names, schema/hash domains,
Skill/Route/deployment identities, idempotency behavior, and fail-closed
cancellation semantics during refactors. Upgrade the current protocol in
place; do not retain a second public v1 transport or consumer mode. Validate
the neutral core separately from each concrete profile,
and do not infer simulator or real-hardware readiness from contract tests. An
in-memory lower-owner loop proves wiring and fail-HOLD semantics only; it is not
evidence that the production launcher supplies authoritative owner identities,
that a safety envelope matches the deployed cell, or that any real motion was
executed safely.
