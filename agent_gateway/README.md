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
  its motion, observation, perception Artifact, policy-Skill, and CubeLift
  bindings.
  `profiles/real` owns the default-off real TCP-pose observation profile and
  the 6+3 primitive application. The preferred deployment shares one mailbox
  with an independent lower Skill owner under `teleop_runtime.skill_deploy`;
  the historic RealTeleop attachment is compatibility-only. Gateway workers
  perform no device I/O, IK, safety filtering, dispatch, or direct HOLD.

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

There is no RGB- or depth-image method in the protocol. A capture Skill declares
its selection parameters, output Artifact slots with media/schema/size
constraints, and a normal executor binding. The same machinery supports point
clouds, logs, calibration artifacts, or other finite Skill input/output without
changing transport semantics. Multi-file results use multiple bindings or one
explicitly encoded archive; v2 does not add a second bundle abstraction.

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

Safety-envelope selection is a motion-admission concern. A Route may publish a
finite set of compatible reviewed envelopes, but every admitted operation pins
one exact envelope ID, content hash, and constraint-set hash. If exactly one
compatible envelope exists, omission retains the single-envelope compatibility
default. If more than one exists and the session advertises
`safety_profile_selection_v1`, omission fails closed with
`safety_profile_selection_required`; the Gateway never guesses a tier from
ordering or names. Passive read and lifecycle Routes do not acquire a motion
envelope through this mechanism.

## Entrypoints

Operator-facing startup, shutdown, risk levels, and remote-developer setup are
maintained in the canonical [`docs/agent_gateway/`](../docs/agent_gateway/README.md)
domain. This package README remains the technical protocol and profile
reference.

The canonical MuJoCo entrypoint is:

```bash
DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint
)"

python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config /absolute/task_config.json \
  --artifact-root /absolute/private/artifact_cache
```

`allocate-unix-endpoint` creates a unique mode-0700 directory under
`<DexTeleop>/temp/ag/` and prints its absent `g.sock` pathname. This separates
the bounded transport namespace from longer history, Artifact, Experience and
log paths. Use `--format json` when the allocation metadata should be retained
alongside a run record.

The allocated endpoint remains explicit: pass the exact returned path to both
the lower Gateway and RPent; neither side discovers or silently relocates it.
Linux limits pathname-based Unix sockets to 107 encoded bytes. Both MuJoCo and
real launchers retain a final validation before constructing a simulator,
device, or lower owner, and the server repeats it immediately before bind. An
advanced caller may pass `--root`, but is then responsible for choosing a root
that leaves enough pathname budget.

`--artifact-root` is optional and default-off. Supplying a real,
non-world-writable directory enables the read-only
`perception.capture_depth` and `perception.capture_rgb` Skills. Depth returns
role-selected native-resolution float32 metric data; RGB returns one
role-selected JPEG per output Artifact ordinal from a single scene cut. The
trusted MuJoCo profile maps semantic roles to lower camera sources. Inline
results contain only bounded metadata and content hashes. Pixel arrays never
enter the control frame.

`python -m control_station agent-gateway-mujoco ...` remains a compatibility
forwarder. It creates no Gateway implementation or runtime state inside
`control_station`.

### Read-only SSH attachment

The canonical host-administration and remote-developer procedure is
[`docs/agent_gateway/SSH_READ_ONLY.md`](../docs/agent_gateway/SSH_READ_ONLY.md).
The details below define the implementation contract rather than a second run
recipe.

`ssh-readonly-proxy` is a small forced-command bridge to an already-running
host-local Unix Gateway. It does not start hardware, create another Gateway,
or own a device. Its fixed method allowlist permits discovery, bounded status,
strictly read-only Skill submission, own-operation query, read-only cancel, and
Artifact downloads. Generic Skill submission, lifecycle, HOLD, Artifact upload,
and all motion routes are absent from the allowlist. The Gateway independently
checks that `submit_read_only_skill_request` resolves to a `t0_read_only`,
`cancel_only`, non-motion Route before admission.

The server-side command is:

```bash
python -m agent_gateway ssh-readonly-proxy \
  --socket-path /absolute/private/gateway.sock
```

Deploy it through a dedicated no-shell SSH account and a key-specific forced
command. Each concurrently attaching developer needs one distinct Unix UID and
one distinct Gateway `client_id`; do not multiplex several developers through
one account or reuse the primary trusted client's identity. Add every remote
identity explicitly when starting the existing Gateway:

```bash
python -m agent_gateway mujoco \
  --socket-path "${DEXTELEOP_GATEWAY_SOCKET}" \
  --task-config /absolute/task_config.json \
  --client-id rpent.dexteleop \
  --client-uid "$(id -u)" \
  --ssh-read-only-client rpent.remote.alice 1101 \
  --ssh-read-only-client rpent.remote.bob 1102
```

`real-rm65` accepts the same repeatable option. With no additional identity,
the endpoint stays mode `0600`. Adding one or more read-only identities changes
only the endpoint mode to `0660`; the Gateway process and the no-shell accounts
must therefore share one dedicated local group. Allocate a traversable session
directory below a pre-created setgid root owned by that group:

```bash
install -d -o gateway-user -g dexteleop-agent-ro -m 2750 \
  /run/dexteleop-agent-gateway

DEXTELEOP_GATEWAY_SOCKET="$(
  python -m agent_gateway allocate-unix-endpoint \
    --root /run/dexteleop-agent-gateway \
    --group-access
)"
```

The allocated session directory is mode `2750`, so the socket inherits the
dedicated group; group users may traverse it but cannot unlink the socket. The
launcher does not change account/group
membership or ownership and does not run `chgrp`. Account and group setup stays
an explicit host-administration step.

The primary `--client-id` remains the only identity eligible for the real
profile's reviewed one-shot motion grant. Additional identities are marked
read-only in the Gateway itself: generic submission, lifecycle, HOLD, event
replay, Artifact upload, and materialization acknowledgement are rejected even
if a caller bypasses the proxy. Strict read-only Skills, bounded status,
own-operation reads/cancel, and own-operation Artifact downloads remain
available. This keeps remote access read-only without introducing a second
Gateway, broker, lease service, or hardware owner.

The SSH identity maps to the Unix UID accepted by the Gateway, while payload
`client_id` remains non-authoritative. The proxy uses the existing
length-prefixed `dexteleop.agent_gateway.unix_json.v2` frames, so a client does
not need a second hardware protocol. Remote Artifact access is limited to
`chunked_rpc` bytes exposed by an own-operation response on the same SSH
connection; arbitrary cache IDs and host-local paths are denied. If another
operation or lower resource claim occupies the device, the normal Gateway
admission/status result is returned; remote access does not queue, preempt, or
acquire a separate lease.

Two additive discovery calls expose a profile-owned, redacted view:

- `get_hardware_resource_views`: `readable`/`drivable` capability plus exact
  semantic Skill references;
- `get_hardware_status_views`: binary `free`/`occupied` occupancy, separate
  health, and an optional active operation projection.

These views are not a registry or lease system. Concrete profiles bind stable
resource IDs to existing Route hashes and lower-owned control-claim and health
reads; the runtime adds its current operation projection. `occupied` means that
an exclusive lower control/resource claim or an active exclusive operation is
present. It does not mean that a Skill is currently moving the robot. READY,
idle, and lower-confirmed HOLD do not release a claim and therefore never imply
`free`; the optional `active_operation_id` reports execution activity
separately. A status read failure is reported fail-closed as unavailable and
occupied. If a profile provides no binding, discovery returns `enabled=false`
and an empty list rather than guessing hardware identity. Gateway admission,
not this advisory view, remains authoritative for whether a Skill may start.

The exclusive real primitive profile publishes its existing arm `component_id`
as one readable/drivable `robot_arm` ownership unit. Its occupancy comes from
the lower `RealAgentMotionOwnerCut`, including READY and HOLD, while health uses
only cached lower state and freshness. When real RGB/depth capture is enabled,
the profile additionally publishes one shared, read-only
`perception.cameras` bundle. A running lower camera worker projects the bundle
as occupied even when no capture Skill is active; camera lifecycle health is
read without capturing a frame. This grouping matches current lower admission
granularity and does not claim independent per-camera leases that the lower
runtime does not provide.

The MuJoCo TCP-pose profile publishes one readable/drivable
`simulation.runtime` resource. Motion, TCP state/history, and optional RGB/depth
capture all share the same scene and runtime owner, so the profile does not
invent independent simulated arm or camera leases. Before the lower runtime is
started the resource is `free` and `unavailable`; after observation-ready
startup it remains `occupied` through READY and lower-confirmed HOLD until the
application closes. Health is derived from the existing execution-gate state
without stepping physics or rendering a frame, and active Skill execution is
still reported separately through the operation projection.

The dedicated independent RM65 Skill entrypoint is:

```bash
python -m agent_gateway real-rm65 \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/right_skill_envelope.json \
  --authorized-side right \
  --authorized-action forward \
  --confirm AUTHORIZE_ONE_REAL_TCP_PRIMITIVE
```

Real RGB/Depth support is opt-in on that same v2 session:

```bash
python -m agent_gateway real-rm65 \
  --socket-path /absolute/private/gateway.sock \
  --arm-ip-assignment /absolute/rm65_arm_ips.json \
  --left-tool-profile /absolute/left_tool.json \
  --right-tool-profile /absolute/right_tool.json \
  --envelope-profile /absolute/right_skill_envelope.json \
  --camera-config /absolute/DexTeleop/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/DexTeleop/robot_motion/configs/rm65_camera_registry.json \
  --artifact-root /absolute/private/artifact_cache \
  --authorized-side right \
  --authorized-action forward \
  --confirm AUTHORIZE_ONE_REAL_TCP_PRIMITIVE
```

For passive camera input without constructing or connecting either RM65 backend,
use the separate camera-only profile:

```bash
python -m agent_gateway real-perception \
  --socket-path /absolute/private/gateway.sock \
  --camera-config /absolute/DexTeleop/teleop_runtime/configs/devices/agent_gateway_realsense.json \
  --camera-registry /absolute/DexTeleop/robot_motion/configs/rm65_camera_registry.json \
  --artifact-root /absolute/private/artifact_cache
```

`real-perception` exposes exactly `perception.capture_rgb` and
`perception.capture_depth` in `real_observe` mode. It owns the configured
RealSense inventory and transient Artifact store, starts and admits cameras
before publishing the Unix socket, and has no RM65 backend, motion Route,
operator Grant, command owner, dispatch path, or domain HOLD path. It therefore
does not accept `--arm-ip-assignment`, tool/envelope profiles, an authorized
motion action, or `--confirm`. The selected collision-scene profile is read only
to derive fixed arm-base geometry for static camera poses; it does not construct
or connect a robot backend.

The three perception flags are all-or-none. The checked-in deployment profile
publishes RGB for `main`, `wrist_left`, and `wrist_right`, but publishes depth
only for `main`. The front camera has a reviewed static camera-to-arm-base
transform; wrist depth is deliberately unavailable because those cameras move
with link 6 and the current registry does not provide the activation-time
joint/FK pose needed by the depth result contract. Hardware reset is disabled.

Both capture Routes are `inspection` / `read_only` / `t0_read_only`: invoking
them needs no motion Grant and cannot call command admission, dispatch, or
HOLD. RGB is JPEG; depth is native-resolution float32 metric NPY with aligned
color intrinsics and the trusted world pose. Payload bytes are committed to
the same generic transient Artifact CAS and retrieved through `chunked_rpc` or
a reviewed same-host `local_path` grant.

For both real profiles, the selected `--collision-scene-profile` is the source
of Station geometry. Its scene URDF fixed-joint chain defines both arm bases in
`dual_arm_solver_world`; the camera registry then defines each fixed camera
relative to its parent link. A Vive/teleoperation `TeleopCalibration` is neither
loaded nor accepted by either entrypoint.

`real-rm65` remains the combined real application. Starting it acquires and
admits all configured cameras, then starts the exclusive dual-RM65 lower owner
before publishing the socket; consequently the existing one-use motion
confirmation and real-hardware safety preflight remain mandatory even when an
upper client intends to invoke only passive Skills. Its runtime TCP-envelope
anchors come from the fresh lower-measured qpos/FK cut after startup HOLD, not
from a teleoperation Home record. A capture invocation itself has no downstream
control effect. Repository offline tests establish neither live camera nor
robot execution for either profile.

The CLI is a concrete lower deployment leaf, not an Agent API. All files and
the selected side/action are trusted operator inputs. The process exposes no
transport until optional cameras have passed consecutive-frame freshness
admission, both RM65 backends have started, lower HOLD is confirmed, a fresh
measured dual-arm cut has seeded the coupled solver, and the Active pre-IK
envelope has been anchored. The exact confirmation authorizes at most one
matching motion request; it does not bypass per-request Gateway admission and
is not consumed by a passive capture.

The MuJoCo launcher starts no RPent planner and connects no real device. Its
socket parent must already exist, resolve to a real non-world-writable
directory, and contain no existing endpoint at the selected path.

The real CLI resolves an `ExclusiveRM65SkillDeploymentProfile`, constructs the
single lower graph through `assemble_exclusive_rm65_skill_lower`, and binds it
to the existing Skill/Gateway contracts through
`assemble_exclusive_rm65_agent_primitive_composition`. It reuses the shared
RM65 backend, coupled solver, force guard, target-envelope, one-use Grant, and
Unix transport implementations without importing or constructing RealTeleop,
VIVE, collection, GUI, or `control_composition`. The application owns the sole
lower control thread and orders shutdown as Gateway fence -> future-cycle
fence -> lower-confirmed HOLD -> control-thread join -> resource close.

The current evidence is offline construction and lifecycle validation, not
real-hardware proof. `attach_real_agent_primitive_gateway` and the RealTeleop
`--agent-primitive-skill-only` option remain compatibility paths; the
independent launcher is the preferred path. `one_shot_client` is only an
authenticated one-request
Unix client and never initializes or dispatches hardware itself.

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
or safety-envelope facts. Those facts come from the selected lower owner and
its Active final pre-IK TCP envelope. Ordinary RealTeleop invocations do not
advertise or start the Gateway, and the repository's offline tests are not
real-hardware validation.

### Real Skill target-frame invariant

The real Skill API is TCP-based, but the RM65 coupled solver is endpoint-frame
based. `RealAgentMotionActivation.target_pose_world`, measured cuts, and the
source value recorded in `BimanualPoseTargetSource` are tool-TCP poses in the
selected execution/world frame. In contrast,
`BimanualPoseTarget.target_pose_by_side` is consumed as a solver-world endpoint
target. Current DMTAC tool profiles commonly introduce roughly `0.14m` of
translation between those frames; the selected profile, not this number, is
the authority.

The only valid path is:

```text
Skill TCP target
  -> lower target owner's solver_world_target_from_tool_tcp_world_target()
  -> BimanualPoseTarget solver endpoint
  -> TCP envelope / IK / dispatch
```

Never pass a TCP pose directly to `BimanualPoseTarget` or duplicate the tool
transform in Gateway/Agent code. A target owner without the conversion method
must be rejected, and tests for this path must use a non-zero offset so an
identity fake cannot hide a frame regression. The canonical action/frame
background is maintained in [`teleop_runtime/README.md`](../teleop_runtime/README.md#action-contract-说明).

Target-independent CubeLift semantics live in `skills/cube_lift`. The MuJoCo
Route, deployment, lifecycle executor, observation binding, evidence, and
terminal evaluator live in `profiles/mujoco/skills/cube_lift`. This keeps task
meaning independent from a simulator while keeping execution policy in the
concrete Gateway profile.

## Observation and visual cuts

Low-rate state observations and pull-based visual captures are separate. RGB is
not a dedicated transport method: `perception.capture_rgb` is an ordinary
`inspection` Skill with a `read_only`, `t0_read_only` Route and generic output
Artifacts. Its deployment profile publishes the trusted semantic roles; the
MuJoCo binding maps `main`, `wrist_left`, and `wrist_right` to current cameras,
renders one scene cut under the existing scene/physics lock, and rejects a cut
if its Station revision changes. Other profiles may bind their own producer to
the same Definition/schema/I/O contract without importing the MuJoCo renderer.
The real binding maps the same roles to an exclusive RealSense inventory,
rejects stale, cross-generation, serial-mismatched, or revision-racing frames,
and commits encoded output through the generic Artifact lifecycle. Device
lifecycle remains outside the passive executor.

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
