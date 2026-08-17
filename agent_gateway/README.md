# Agent Gateway

`agent_gateway` is DexTeleop's top-level, upper-framework-neutral attachment
and Skill execution control plane. It is separate from the Control Station GUI.
An upper framework attaches through the published protocol and transport; it
does not import a simulator, device, controller, or Control Station runtime.

RPent is the first reviewed consumer, not a runtime dependency. Wire requests
must not contain an RPent Task, Program, Planner, Memory, Experience, or upper
SkillAttempt object. Route/deployment identity remains a trusted lower binding,
not caller-authored Agent input.

## Package ownership

- `protocol/`: versioned requests, operations, observations, evaluation
  receipts, Skill/schema/I/O catalogs, Artifact and replay-event contracts,
  Route manifests, a construction template, and pure validators.
- `runtime/`: client access policy, exact executor registry, bounded operation
  store, idempotency/reconciliation, passive reads, lifecycle workers, and
  injected lower operation/HOLD ports. `AgentGatewayProtocolRuntime` binds one
  session's discovery surface, replay journal, and optional Artifact cache and
  transfer broker.
- `transport/`: authenticated bounded transports and the transport-neutral wire
  adapter. The current local transport uses Unix peer credentials.
- `skills/`: target-independent Skill definitions and Skill-specific payload
  schemas. `skills/motion` owns Agent-visible motion identities and schemas;
  pure pose primitives remain lower-owned by `robot_motion.primitives`.
  Skills do not select a concrete route, frame, device, safety envelope, or
  deployment.
- `profiles/`: exact application/deployment bindings and executors. The current
  `profiles/mujoco` package owns the no-model TCP-pose Gateway application and
  its motion, observation, visual snapshot, policy-Skill, and CubeLift bindings.
  `profiles/real` owns the default-off, attach-only real TCP-pose observation
  profile and the 6+3 primitive application. Gateway workers submit immutable
  requests to one lower RealTeleop port; they perform no device I/O, IK, safety
  filtering, dispatch, or direct HOLD.

Shared activation-time application of reviewed TCP primitives lives in
`profiles/motion_primitive_resolution.py`; it derives an absolute lower target
from measured state and an exact frame graph but owns no safety, IK, authority,
or dispatch. MuJoCo-specific 6+3 Route IDs and immutable parameter profiles live
in `profiles/mujoco/motion_primitive_profiles.py`. The historic exports from
`profiles/mujoco/motion_deployment.py` and `motion_skill.py` remain available so
this organization-only split does not break consumers.

The neutral `protocol/runtime/transport` core must not import
`control_station`, concrete simulation, hardware, teleop, model deployments,
or an upper Agent framework. A concrete profile may assemble a Control Station
application and lower target through typed ports, but the lower target remains
the only canonical authority/safety/IK/reference/dispatch owner.

## One v2 Skill protocol

The current and only public Skill protocol is `dexteleop.agent_skill.v2`; the
local framed transport is `dexteleop.agent_gateway.unix_json.v2`. This is an
in-place repository-wide upgrade, not a second protocol family. Existing
inline Skills, including the nine TCP primitives, publish explicit empty
Artifact slot lists and run in an Artifact-disabled v2 session. A deployment
that exposes Artifact-bearing Skills enables the same v2 data plane through
its session capability document.

An attaching client discovers five mutually bound documents before invoking a
Skill: Definition Catalog, Schema Catalog, Skill I/O Catalog, Route Manifest,
and Session Capabilities. Hash closure prevents mixing documents from another
runtime session or deployment. The Unix facade additionally exposes bounded
event replay and generic operations to begin/upload/commit an Artifact,
begin/read/acknowledge a download, and acknowledge an upper-owned durable
materialization.

There is no depth-image method in the protocol. A Skill such as “capture a
specific view's latest depth” declares parameters for the requested view, an
output Artifact slot with media/schema/size constraints, and a normal executor
binding. The same machinery supports point clouds, logs, calibration artifacts,
or other finite Skill input/output without changing transport semantics. Multi-file
results use multiple bindings or one explicitly encoded archive; v2 does not add a
second bundle abstraction.

Large bodies never enter an operation or event payload. `AgentArtifactRef` is
the portable content identity; filesystem paths and other locators exist only
inside short-lived, authenticated access grants. The Gateway's managed store
is a transient, quota-bounded handoff cache:

1. reserve quota and create/recover a server-owned staging inode;
2. upload bounded chunks through `chunked_rpc`; an optional reviewed
   same-host `local_path` grant serves only committed read/download bytes;
3. verify byte length and SHA-256, then atomically commit into the local CAS;
4. pin inputs to the operation and retain outputs through the handoff window;
5. issue session/client-bound read grants and receive transfer receipts;
6. record the upper layer's materialization receipt once it has a separate
   durable copy;
7. remove unpinned data after minimum retention/handoff deadlines, retaining a
   tombstone rather than treating a stale path as identity.

Events use monotonic per-session sequence numbers, bounded retention, and an
explicit `gap_detected` result. They are notification hints only. After a gap,
clients reconcile through authoritative operation and discovery queries.

## Building a Skill

`protocol.skill_registration.SkillRegistrationTemplate` is the recommended
construction seam. A registration supplies one target-independent
`SkillDefinition`, one strict parameter/result/evaluation schema entry, one
`SkillIOContract`, and one or more deployment-independent Routes. The template
checks all identities, schema hashes, parameter profiles, I/O hashes, and
Route closure before `build_skill_registration_catalogs` emits the three
discovery catalogs. Concrete deployment bindings, device acquisition, camera
ownership, safety, and executors remain in `profiles/`; they are deliberately
not hidden inside the template.

Control effect is a closed contract, not a naming convention. An `inspection`
Definition can only use a `read_only` Route; that Route must be
`t0_read_only`, use `cancel_only`, advertise no robot motion or publication
opening, and bind an executor that declares the same effect class. Motion and
policy-task Definitions use motion Routes, while lifecycle Definitions use
lifecycle Routes. Neutral no-I/O test/dry-run ports cannot prove either read or
control execution. The control-domain HOLD port remains separate from passive
read execution.

## Entrypoints

The canonical MuJoCo entrypoint is:

```bash
python -m agent_gateway mujoco \
  --socket-path /absolute/private/gateway.sock \
  --task-config /absolute/task_config.json \
  --artifact-root /absolute/private/artifact_cache
```

`--artifact-root` is optional and default-off. Supplying a real,
non-world-writable directory enables `perception.capture_depth`, which returns
native-resolution float32 metric depth through a normal output Artifact slot.
The inline result contains view/capture/intrinsic metadata; the Artifact body
is an `application/x-npy` array with invalid depth represented as `0.0`.

`python -m control_station agent-gateway-mujoco ...` remains a compatibility
forwarder. It creates no Gateway implementation or runtime state inside
`control_station`.

The MuJoCo launcher starts no RPent planner and connects no real device. Its
socket parent must already exist, resolve to a real non-world-writable
directory, and contain no existing endpoint at the selected path.

The Gateway package still has no direct real-device launcher. A reviewed
composition root calls
`agent_gateway.profiles.real.motion_factory.attach_real_agent_primitive_gateway`
with exact observation/owner identities, frame graph, lower motion-port
configuration, safety-envelope references, and allowed clients. That function
starts no socket, thread, control loop, device, or motion. It returns a new
RealTeleop component aggregate and a Gateway session sharing one lower port;
the caller must run the returned aggregate in the existing RealTeleop owner
loop before it may expose a transport. The canonical RealTeleop application now
provides that default-off composition behind
`--agent-primitive-skill-only`; `agent_gateway.profiles.real.one_shot_client`
is only the authenticated one-request Unix client and never initializes or
dispatches hardware itself. The server pre-authorizes exactly one side/action,
atomically binds one one-use Grant to that request, and disables Vive command
fallback for the entire Skill-only process.

## Skill and execution boundary

The default-off MuJoCo profile publishes one lower motion domain with separate
versioned Skills for absolute TCP targets, semantic cardinal steps, and fixed
level orientations.
`motion.move_tcp_to_pose@1` accepts an exact lower target;
`motion.step_tcp_cardinal@1` accepts `forward`, `backward`, `left`, `right`,
`up`, or `down`. The cardinal target is resolved from measured TCP state only
when the lower Skill is activated. Distance, frame/tool/calibration identities,
primitive hash, safety envelope, IK, dispatch, evaluation, and HOLD remain
trusted deployment facts.

Target-independent motion semantics and payload schemas live in
`skills/motion`. They also define fixed level-orientation Skills for forward,
forward-left 30 degrees, and forward-right 30 degrees. Together, the cardinal
Skill's six enum choices and the three fixed-orientation Skill identities expose
nine bounded lower primitives. The current single-right-arm no-model launcher
publishes all nine through exact MuJoCo Routes. Cardinal directions and absolute
headings are resolved in the canonical Station frame, then transformed into the
right-arm execution frame; the arm-base frame is not relabelled as canonical
desk axes. Fixed orientations use their own route-scoped safety envelope because
they preserve translation but may require a larger angular correction than a
cardinal step or generic small pose move. Publication still grants no motion by
itself: activation preflight, lower safety, IK, dispatch, evaluation, and HOLD
remain mandatory.

The real profile exposes the same nine primitives through four exact
`t2_real_motion` Routes: one cardinal Route with six bounded direction values
and three fixed-orientation Routes. Every request requires a one-use operator
Grant bound to the complete pre-grant request. Activation resolves the target
from a fresh lower-measured TCP pose, and completion requires distinct fresh
measurement cuts, at least one lower dispatch, the configured stable-arrival
count, and terminal lower-confirmed HOLD. Cancellation, disconnect, timeout,
fault, and Gateway close all use that same lower HOLD boundary.

The real Gateway does not treat the existing foot switches as an Agent
deadman, and it cannot manufacture owner epochs, frame/calibration identities,
or safety-envelope facts. Those facts come from the attached lower RealTeleop
owner and its Active final pre-IK TCP envelope. Ordinary RealTeleop invocations
do not advertise or start the Gateway, and the repository's offline tests are
not real-hardware validation.

Target-independent CubeLift semantics live in `skills/cube_lift`. The MuJoCo
Route, deployment, lifecycle executor, observation binding, evidence, and
terminal evaluator live in `profiles/mujoco/skills/cube_lift`. This keeps task
meaning independent from a simulator while keeping execution policy in the
concrete Gateway profile.

## Observation and visual cuts

Low-rate state observations and pull-based visual captures are separate. A
deployment publishes its complete current RGB source catalog and implements one
bounded capture request; the request grants no streaming or motion authority.
The MuJoCo profile currently publishes `main`, `wrist_left`, and `wrist_right`,
renders under the existing scene/physics lock, and rejects a cut if the bound
Station revision changes across capture. Other profiles bind their own producer
to the same transport-neutral facade and do not import the MuJoCo renderer.

## Compatibility

Existing imports under `control_station.agent_gateway`,
`control_station.sim_hil_profile.agent`, `control_station.tasks.cube_lift`, and
`control_station.sim_hil_profile.tasks.cube_lift`, plus the historic real
observation Gateway module, are thin re-export shims.
`agent_gateway.protocol.motion_schemas` is also a temporary compatibility
re-export; new motion-Skill code imports `agent_gateway.skills.motion`.
New code must import the canonical `agent_gateway.*` modules. Compatibility
modules may be removed only after production, test, documentation, and external
consumer audits establish that their callers have migrated.

Some immutable Skill/Route identifiers and hash-domain strings still contain
the historical `control-station` spelling. They are protocol identities, not
current Python ownership claims, and remain unchanged so the structural move
does not invalidate manifests, schema snapshots, idempotency records, or
external reconciliation.
