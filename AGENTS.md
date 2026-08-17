# DexTeleop Project Rules

DexTeleop contains shared robot motion, real-robot teleoperation, data
collection, policy deployment, simulation, and dataset contracts. Keep the
directory boundaries stable and do not add a new production entrypoint without
checking the existing README and architecture plan.

## Canonical entrypoints

- GUI human teleop and formal collection:
  `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui`
- Headless/compatibility collection:
  `python -m teleop_runtime.collection`
- Policy deployment:
  `python -m teleop_runtime.deploy`

`teleop_runtime.collection` is headless/compatibility only. `teleop_runtime/core/kernel.py`
is experimental.

## Directory ownership

- `control_composition/`: reusable, backend-neutral multi-source control
  composition, lifecycle reducers, causal lineage, and non-authoritative
  difference/feedback. It may depend on `robot_motion` and pure data contracts,
  but must not import concrete sim, hardware, teleop, deployment, or
  `control_station` modules. It owns no authority, canonical safety, execution,
  device/model worker, replay cursor, or episode lifecycle.
- `control_station/`: GUI-facing application control plane. It owns typed
  operator intents, rebuildable UI state, application lifecycle/profile
  coordination, and thin source/target bindings. It may construct lower
  implementations and connect them through typed ports, but reusable source
  and control semantics remain lower-owned: workers and source admission,
  queues/cursors, control/data processing and journals, execution/persistence,
  safety, and dispatch. Compatibility/profile shims may re-export APIs, bind
  defaults, or translate application intents/status, but must not duplicate
  lower implementations or state. `control_station` may import active lower
  packages; lower packages must not import it.
  Migration exceptions must remain explicit in boundary tests or documentation.
- `agent_gateway/`: upper-framework-neutral external Skill control plane. Its
  core `protocol/`, `runtime/`, and `transport/` packages own versioned wire
  contracts, authenticated attachment, request admission/reconciliation,
  bounded operation projection, and exact executor registration. Target-
  independent Skill definitions live under `skills/`; concrete application,
  simulator, observation, executor, and deployment bindings live under
  `profiles/`. The Gateway must not import or embed an upper planner, Task,
  Program, Memory, Experience, or framework-specific adapter, and it owns no
  canonical authority, safety, IK, reference, dispatch, or episode lifecycle.
  Core packages must not import `control_station`, concrete simulation,
  hardware, teleop, or model deployments. Concrete profiles may assemble
  application leaves through typed ports. Historic `control_station` import
  and CLI paths are forwarding compatibility shims only and must not regain
  implementation or runtime state.
- `robot_motion/`: shared motion semantics, Rig, Safety, Planner, position
  control, and runtime contracts; no vendor-device initialization or training.
- `hardware/`: vendor SDKs and device backends; no dataset schema or learner.
- `sim/`: simulation-only backends, tasks, and adapters; may use `robot_motion`,
  but must not depend on `teleop_runtime` or real hardware.
- `collection_runtime/`: device-, control-, and UI-neutral live collection
  lifecycle and persistence mechanics. It may depend on `dataset_contracts`,
  `dataset_interface`, and pure domain contracts, but must not import
  `teleop_runtime`, `control_station`, concrete hardware, simulation, or GUI
  modules. It owns no device production, robot control, operator intent, or
  dataset schema definition.
- `teleop_runtime/`: real-robot teleop, GUI/headless orchestration, source-bound
  collection adapters, compatibility collection paths, and policy deployment
  frontends. Compatibility shims may re-export migrated collection runtime
  APIs but must not duplicate their implementations.
- `deployments/`: model client, protocol, and artifact integration.
- `dataset_contracts/`: pure schemas, identities, hash domains, and validation
  contracts; no runtime, device, GUI, or control ownership.
- `dataset_interface/`: data adaptation, projection, reader, window/batch, and
  validation interfaces. It may depend on `dataset_contracts` and pure domain
  contracts, but must not become a device/control runtime or authoritative
  episode owner, and must not import `control_station`.
- `experiments/`: versioned, reproducible, non-canonical hypothesis
  implementations and offline evaluation. It may consume canonical public
  contracts, but production packages must not import it. It owns no canonical
  schema, persistent data semantics, artifact integration, authority, safety,
  dispatch, resource lifecycle, or production entrypoint. Reusable logic is
  promoted to its production owner rather than imported from this tree.
- `model/`: local policy weights, checkpoints, runtime sidecars, and explicitly
  pinned external source boundaries only. It is not a Python package or an
  AgenticRobot source-code owner; versioned artifact contracts and integration
  remain under `deployments/`. `model/openpi` is the independently owned OpenPI
  Git submodule: its gitlink is the source-version boundary, it keeps its own
  Python/runtime environment, and AgenticRobot code must not treat it as the
  `model.openpi` package or vendor its generated/runtime assets.
- `legacy/`: confirmed-archived code/docs only; production code must not import it.
- `tests/`: verification, never runtime code. Root `todos/` is Git-ignored local planning state;
  tracked code, tests, and documentation must not import or link to it. Plans that require shared
  review belong under the owning `docs/**/plans/` directory.

Shared code should depend toward `control_composition`, `robot_motion`, and pure
data contracts. `control_composition` depends downward on `robot_motion`; the
reverse dependency is forbidden. `sim` must not reverse-depend on
`teleop_runtime`. `control_station` and concrete `agent_gateway` profiles are
application leaves; lower-level packages must not reverse-depend on either.
The neutral Gateway core must remain independent of concrete application
profiles.

Ownership follows reusable domain semantics and resource lifecycles, not import
legality or the current caller.

For one logical runtime resource or stream, source production, cursor or plan
advancement, authority, canonical safety, dispatch, and episode commitment each
have one authoritative owner. Passive bounded observers, fanout queues, and UI
caches are allowed, but they must not mutate or independently reconstruct the
owned state.

## Delivery scope and evidence

- Judge completion against the user-requested outcome. A contract, module,
  fixture, or profile may be complete in its own requested scope, but does not
  prove downstream runtime integration. When broader integration could
  reasonably be inferred, state the actual caller or its absence, the
  environment and furthest boundary exercised, and the next relevant unverified
  boundary. Do not impose a fixed evidence sequence.
- New plans, profiles, adapters, fixtures, wrappers, exports, or modules are not
  progress by themselves. Add one only when requested or needed for the current
  implementation, compatibility or rollback, or relevant verification. Describe
  structural progress by the responsibility, dependency, or call edge changed,
  not merely by files moved, abstractions added, or test counts.

## Concurrent worktree coordination

These rules apply when linked worktrees may be active concurrently or work will
be handed off for integration. Before substantive edits, inspect every linked
worktree's branch, tip, status including untracked files, and declared scope;
repeat the relevant check before commit, integration, and push. Treat other
worktrees as read-only: never edit, reset, clean, stash, move, or remove them.

For newly created concurrent work, use descriptive names:

- task branch: `task/<owner>/<area>-<slug>-<YYYYMMDD>`;
- integration branch: `integration/<owner>/<batch>-<YYYYMMDD>`;
- recovery branch: `recovery/<owner>/<source>-<YYYYMMDD>`;
- worktree directory: include owner, scope, and date; keep it short and place it
  under an approved ignored parent or outside the primary checkout.

Do not create a worktree when one already owns the logical task; an additional
one needs a distinct scope or isolation reason. Legacy names need not change,
and names never prove ownership, readiness, or freshness. Mark completed work
`superseded`; remove it only with owner or user authorization after preserving
or explicitly abandoning unique commits.

Every active concurrent branch must declare owner, state, base, write scope,
semantic/public-contract scope, dependencies, and validation; immutable states
also pin the branch tip, and integration branches list ordered input SHAs. Use
`branch.<name>.description` within one clone and carry the record in a handoff
across clones. Task branches move `active` -> `ready-for-integration`;
integration branches move `integrating` -> `validated-for-mainline`; `blocked`
and `superseded` are not selectable candidate states.

At each coherent phase boundary, validate and commit intended versioned
changes before yielding or handoff. Tracked dirty status means work remains
active and is not integration-ready. If the task continues, keep the branch
`active` but clean and record its checkpoint tip, validation, and next scope;
when complete, use `ready-for-integration`. Never commit generated artifacts,
secrets, caches, or unrelated user work merely to make status clean.

Path separation does not resolve semantic overlap. Give each versioned schema,
public API/export/entrypoint, and ownership or safety contract one active owner;
resolve ordering before dependent work continues. Treat an old-base overlap as
a semantic conflict even when Git merges it textually.

Ready and validated states bind one clean, reviewed commit and its evidence, not
a moving branch name. A new commit or relevant working-tree change invalidates
that state. Preserve identified runtime artifacts; never delete them to produce
a clean status.

During an integration batch, one visibly designated mainline integrator alone
may move `main`; unclear or competing ownership blocks main movement. Integrate
only pinned inputs on an isolated branch from pinned main, test the committed
result, then recheck clean main plus local/remote main and input pins. Only
fast-forward main to the validated commit; never hold an uncommitted merge on
main during long tests. If main or an input moves, rebuild and rerun affected
validation. Pause when unresolved overlap changes public semantics, ownership,
safety, persisted data, or expected behavior.

## Safety and change protocol

- Prefer read-only, dry-run, mock, contract, and `--help` checks.
- Do not start real devices or execute motion, homing, replay, servo,
  force-control, reboot, or other device-state changes without explicit scope
  and the relevant safety check.
- Remote hardware work must use the wrappers and rules in `codex_remote/`.
- Inspect status/diff first, preserve concurrent work, keep edits scoped, and
  do not reset or overwrite unrelated changes.
- Before any dataset transform that combines hardlinks, partial selection, or
  overwrite, verify source/target device, inode, link count, and independent
  backup receipts. A differently named path is not an independent backup.
- Dataset video transforms must write a fresh temporary inode, validate the
  declared/decoded frame count and tail decode, then atomically replace the
  target. Stop at the first decoder warning or count mismatch; never accept
  process exit status alone as data-integrity evidence.
- Ask for clarification when an unresolved choice affects safety, public
  control/data semantics, ownership, or destructive behavior; otherwise state
  assumptions and proceed.
- For any stateful control modifier, bound the final composed output including
  persistent terms at the last authoritative boundary. Component and rate
  limits are not total-action limits; persistent feedback must use an explicit
  incremental baseline rather than an absolute target-to-command residual.
- Shadow evidence cannot authorize Active-only feedback or state transitions;
  exercise them offline in closed loop, including nonzero nominal residuals,
  before any real-motion command is provided.
- After unexpected real motion, freeze that Active path and preserve its exact
  command, configuration, logs, and process evidence. Diagnose read-only and do
  not tune or retry until both the cause and the missing safety gate are closed.
- Real Agent motion has an explicit frame boundary: public Skill/observation
  poses are `tool_tcp`, while the RM65 coupled solver consumes its configured
  solver endpoint frame. The current DMTAC profiles commonly have about
  `0.14m` of TCP offset, but the selected tool profile is authoritative. Never
  pass a TCP pose directly to `BimanualPoseTarget`/IK or recreate the tool
  transform above the lower target owner; require the canonical lower conversion
  and fail closed when it is unavailable.
- Use `AGENT_TEMP_NOTE:` only for active collaboration and remove obsolete notes.

## Verification and environment metadata

Run the smallest relevant tests and report them. Use `git diff --check`; add
targeted tests, `py_compile`, and entrypoint `--help` checks as appropriate.
Hardware tests and remote runs require explicit scope.

Run repository tests through `bash scripts/run_pytest.sh ...`. The wrapper
selects the supported `dexteleop310` Python 3.10 environment, verifies the
required `osqp` test dependency, disables ambient pytest plugins, and allocates
a process-unique short `--basetemp` under `/tmp`. Do not run DexTeleop tests
with the RPent Python environment and do not place pytest `--basetemp` inside a
worktree.

AF_UNIX tests must request the shared `short_unix_socket_path_factory` fixture;
never derive a live socket endpoint from `tmp_path`, the repository path, or a
worktree-local `temp/` directory. Production endpoints must continue to reject
overlong paths rather than silently relocating them. The `/tmp` exception is
test-only and does not change runtime artifact placement.

Only dependency, Python-version, package-discovery, or installation changes
require updates to `pyproject.toml`, `environment.yml`, `uv.lock`,
`scripts/install_conda_env.sh`, or `README_ENV.md` as applicable. Source-only
and documentation changes do not require lock updates. Never hand-edit
`uv.lock`; regenerate it after dependency changes.

## Legacy policy

Move code to `legacy/` only after confirming there are no production, import,
test, or documentation callers, a replacement exists, and a migration note
records the reason, replacement, and deletion condition. Keep required
compatibility shims in their active import paths.
