# Embodiment runtime provider contract

This document describes the current implementation boundary for adding a
target embodiment without coupling shared control, teleoperation, collection,
deployment, or dataset code to a vendor SDK. It is an interface contract, not
evidence that a second embodiment or the existing RM65 production path has
been migrated.

## Implemented boundary

Application selection and target-facing integration use five layers:

1. `robot_motion.target_runtime` owns backend-neutral provider, manifest,
   capability, operation, observation, status, handle, and binding contracts.
   It does not import hardware, simulation, teleoperation, deployment, or
   `control_station` implementations.
2. `TargetRuntimeProfileSpec` is the portable, serializable application
   allowlist. It pins the provider identity, embodiment definition, component
   action schema, target capability, operations, observations, assets, and
   supported clocks. It contains no Python loader or module path.
3. `TargetRuntimeProfileBinding` pairs one spec with an explicit allowlisted
   provider loader. `TargetRuntimeProfileCatalog` loads only the selected
   provider and performs an offline manifest preflight. Importing or listing
   profiles does not discover integrations or import vendor code.
4. `prepare_target_runtime_handle` re-resolves the selected configuration,
   requires the exact preflight manifest, and may construct only a provider
   declared `inert_handle`. Construction remains a pending transactional
   acquisition while the registry validates the handle and the application
   reads and validates its initial `created/not ready` status. Every failure
   before explicit commitment invokes provider-owned construction rollback.
   Preparation never calls an operation or execution port and does not grant
   authority.
5. `EmbodimentApplicationSuiteSpec` sits above target profiles and pins one
   complete embodiment integration package, its assets, and its complete
   application surfaces. `EmbodimentApplicationSuiteBinding` pairs those
   surfaces with explicit lazy loaders. The catalog loads only one selected
   surface and returns it as an opaque implementation; it does not split,
   instantiate, or invoke the embodiment's algorithms.

The application owns profile selection and the lifetime of the returned
aggregate. The concrete provider owns target construction, its ports, and its
device-specific lifecycle implementation. Canonical authority, safety, permit
issuance/consumption, action adaptation, and dispatch remain with their
existing lower owners.

### Application module layout

The canonical application imports are now grouped under
`teleop_runtime.target_runtime`:

- `profiles` owns portable profile specs, allowlisted bindings, catalog
  selection, and offline preflight;
- `preparation` owns the transactional handoff from exact preflight to an
  inert handle acquisition;
- `application_suites` owns portable whole-integration profiles, explicit
  surface-loader bindings, and selected-only loading of opaque application
  implementations;
- `integrations` is an explicit, non-discovering namespace for concrete
  application adapters. Importing `teleop_runtime.target_runtime` does not
  import an integration.

The former `teleop_runtime.target_profiles`,
`teleop_runtime.target_runtime_preparation`, and
`teleop_runtime.deploy.rm65_target_runtime_provider` paths are compatibility
re-export shims. They construct no registry or runtime state and preserve
object identity with the canonical implementations. The RM65 candidate
adapter is application-level compatibility projection rather than deployment
ownership. Its frozen target capability/manifest and backend execution
evidence live with the RM65 arm backend contracts under
`teleop_runtime.backends.arms`; deploy paths remain compatibility exports.
This prevents RM65 backend and diagnostic modules from importing the policy
deployment frontend merely to share target facts.

### Whole-integration boundary

An embodiment application surface is a complete vertical implementation. It
may own embodiment-specific teleoperation algorithms, retargeting, IK,
collision handling, control and recovery policy, state reading, dispatch,
calibration, assets, collection adapters, and deployment adapters. The shared
suite catalog identifies and selects that complete implementation; it does not
require those internals to implement a common algorithm or component layout.
The complete surface is the replacement unit: the framework does not
independently select its IK, retargeting, collision, recovery, or dispatch
algorithm.

RM65's current real-teleop implementation is intentionally RM65-specific. Its
dual-arm optimization, collision scene, recovery behavior, force-control
paths, and component aggregate are not candidates for mechanical extraction
merely to satisfy a generic builder signature. Code moves into a shared layer
only after multiple embodiments demonstrate the same domain semantics and
resource ownership.

## What an embodiment integration must provide

An integration package that will eventually be selectable must provide:

- a stable `TargetRuntimeProviderIdentity`, including its API, implementation
  and dependency identities and either `preflight_only` or `inert_handle`
  construction mode;
- a side-effect-free manifest resolver that binds every handle-affecting
  choice to a `ResolvedTargetRuntimeManifest`;
- exact component-keyed action and target capabilities, including payload
  schemas, frames, units, resources, and allowed active combinations;
- observation and lifecycle/device operation capabilities when supported;
- content identities for required assets such as robot descriptions, meshes,
  controller maps, tool profiles, and calibration schemas;
- for an `inert_handle` provider, one handle whose execution, status,
  observation, and operation ports all expose the same resolved manifest and
  whose construction does not connect, start, home, command, or activate a
  target;
- for an `inert_handle` provider, an idempotent
  `rollback_target_runtime_handle_construction` implementation that releases
  resources associated with any returned, uncommitted handle without
  submitting execution work or invoking target operations. It must tolerate a
  handle that fails structural or initial-status validation;
- internal rollback for partial construction that raises before returning a
  handle, plus provider-specific committed lifecycle/cleanup behavior. The
  generic registry cannot roll back a handle it never received and cannot
  invent a safe close operation for an unknown active target.

Provider configuration is deliberately opaque to the shared contract, but it
is not outside integrity checks: any configuration value that can change the
future target identity, semantics, operations, observations, or assets must
change the resolved manifest. Preparation detects a different manifest before
creating the handle.

## Switching and application composition

Selection is by a stable application suite and surface ID, not by importing a
module path from configuration and not by replacing a URDF in an existing
builder.
A safe future switch has these phases:

```text
portable application-suite and surface selection
  -> explicit allowlisted surface loader
  -> complete embodiment-specific application implementation
  -> that surface's exact target-profile selection
  -> allowlisted provider binding
  -> side-effect-free manifest preflight
  -> pending inert-handle acquisition and validation
  -> explicit commit to one lifecycle owner
  -> application/lower-owner lifecycle and authority gates
  -> execution
```

The returned `TargetRuntimeHandlePreparation` is still uncommitted. A caller
must explicitly call `commit_runtime_handle` when it is ready to become the
single lifecycle owner, or call `rollback_uncommitted_handle`; using the
preparation as a context manager automatically rolls back if it exits while
still pending. Commitment is an ownership handoff only. It does not start a
target, invoke an operation, submit execution, or grant authority. Once
committed, provider construction rollback is no longer legal.

Switching an active target is not implemented by the catalog. A pending target
may be abandoned only through construction rollback. A committed current
target must first complete its declared, permitted hold/close lifecycle and
release its resources through its provider-owned implementation. Only then may
the application prepare another target. A profile hash match proves contract
identity only; it does not prove local asset presence, fresh state, hardware
readiness, safety acceptance, or motion authorization.

An application suite may package its own source devices, teleoperation
mapping, and algorithms. Reuse through source/retarget adapters remains
possible when semantics genuinely match, but is not a framework requirement.
Dataset capture remains bound to pure embodiment and stream profiles in
`dataset_contracts`/`dataset_interface`; runtime objects and SDKs must not
enter dataset schemas.

## Compatibility status

The RM65 policy-arm-qpos candidate is currently an exact `preflight_only`
compatibility provider. Its frozen manifest/evidence hashes and existing
formal entrypoints are unchanged. It cannot create a target-runtime handle.

No second real embodiment, simulated embodiment, SDK adapter, production
builder migration, or motion path is introduced by this framework work.
Application-suite fixtures cover multiple complete surfaces with different
target profiles, portable serialization, asset closure, selected-only lazy
loading, opaque implementations, and loaded-identity drift. They do not claim
that any current RM65 surface has been bound to a suite.
Arbitrary-topology synthetic fixtures cover a single 7-DoF manipulator plus an
independent 1-DoF tool, nonempty observation/operation/asset surfaces, profile
serialization, selected-only loading, configuration drift, invalid handles,
initial-status failure, transactional commit/rollback, rollback failure, and
preflight-only rejection. These fixtures validate interface shape and
fail-closed behavior; they are not downstream runtime or hardware acceptance.

The formal RM65 real-teleop and collection builders remain compatibility code.
They directly own RM65/VIVE/Inspire/DM-Tac calibration, IK, collision,
renderer, recovery, and hardware policy and remain one RM65 vertical
integration. They must not be relabeled as generic or decomposed merely by
wrapping their constructor in a profile. A future application composition
root may select that complete surface after an exact suite binding exists;
the surface's internal control logic remains RM65-owned.
