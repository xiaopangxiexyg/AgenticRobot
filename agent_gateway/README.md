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
  receipts, Skill/schema catalogs, Route manifests, and pure validators.
- `runtime/`: client access policy, exact executor registry, bounded operation
  store, idempotency/reconciliation, passive reads, lifecycle workers, and
  injected lower operation/HOLD ports.
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
  profile; it consumes immutable owner publications and performs no device or
  motion I/O.

The neutral `protocol/runtime/transport` core must not import
`control_station`, concrete simulation, hardware, teleop, model deployments,
or an upper Agent framework. A concrete profile may assemble a Control Station
application and lower target through typed ports, but the lower target remains
the only canonical authority/safety/IK/reference/dispatch owner.

## Entrypoints

The canonical MuJoCo entrypoint is:

```bash
python -m agent_gateway mujoco \
  --socket-path /absolute/private/gateway.sock \
  --task-config /absolute/task_config.json
```

`python -m control_station agent-gateway-mujoco ...` remains a compatibility
forwarder. It creates no Gateway implementation or runtime state inside
`control_station`.

The MuJoCo launcher starts no RPent planner and connects no real device. Its
socket parent must already exist, resolve to a real non-world-writable
directory, and contain no existing endpoint at the selected path.

## Skill and execution boundary

The default-off MuJoCo profile publishes one lower motion domain with separate
versioned Skills for absolute TCP targets and semantic cardinal steps.
`motion.move_tcp_to_pose@1` accepts an exact lower target;
`motion.step_tcp_cardinal@1` accepts `forward`, `backward`, `left`, `right`,
`up`, or `down`. The cardinal target is resolved from measured TCP state only
when the lower Skill is activated. Distance, frame/tool/calibration identities,
primitive hash, safety envelope, IK, dispatch, evaluation, and HOLD remain
trusted deployment facts.

Target-independent motion semantics and payload schemas live in
`skills/motion`. They also define fixed level-orientation Skills for forward,
forward-left 30 degrees, and forward-right 30 degrees. The current
single-right-arm no-model launcher does not publish those Routes because its
frame binding and reviewed safety profile do not establish the required
canonical desk axes and full heading range. A future launcher must publish its
own exact observation, frame, safety-envelope, IK, and dispatch bindings before
making them Active.

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
