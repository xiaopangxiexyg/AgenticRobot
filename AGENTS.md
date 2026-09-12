# AgenticRobot Project Instructions

AgenticRobot contains reusable robot-motion, control, data, simulation,
teleoperation, deployment, and external Agent Gateway packages. Keep durable
instructions here limited to repository-wide boundaries; use the linked
documents and Skills for conditional workflows and implementation detail.

## Sources and entrypoints

- Read `docs/architecture/runtime_domain_boundaries.md` before changing package
  ownership, dependency direction, application assembly, or runtime authority.
- Consult the relevant resource class in `docs/setup/LOCAL_RESOURCES.md` and the
  matching entries in `docs/setup/local_resources_manifest_v1.json` before
  changing ignored resources or their bindings; do not load the whole manifest
  when a focused query is sufficient.
- Locate the relevant `README_ENV.md` heading for Python, dependency,
  installation, or lockfile work; load only that section and its shared boundary.
- Use the existing README and CLI `--help` for runnable entrypoints. Do not add
  a second production entrypoint for an existing lifecycle.

## Conditional workflows

- Use `$change-agent-gateway` for Gateway-owned protocol, runtime, transport,
  semantic Skill, Route, Artifact, compatibility, or profile-binding changes.
- Use `$debug-teleop-ik-motion` for pose-to-command teleoperation, IK,
  reference, interpolation, dispatch, or stateful-modifier faults.
- Use `$manage-project-local-resources` for ignored assets, datasets, model or
  vendor bundles, machine-local configuration, generated output, and
  cross-worktree resource movement.
- Use `$remote-access` when work crosses the local/remote execution boundary for
  inspection, transfer, processes, devices, or hardware.

For a mixed task, use each applicable Skill but keep these ownership boundaries
separate: domain Skills own code and data semantics; `$remote-access` owns only
the remote operation used to inspect, transfer, or execute them.

## Repository boundaries

- Reusable packages own domain semantics and resource lifecycles. Application
  leaves such as `control_station` and concrete `agent_gateway` profiles may
  assemble lower packages through typed ports; lower packages must not
  reverse-depend on those leaves.
- `agent_gateway` core remains upper-framework- and backend-neutral. Target
  facts and executor construction belong in concrete profiles.
- For one resource or stream, source production, cursor or plan advancement,
  authority, canonical safety, dispatch, and episode commitment each have one
  authoritative owner. Observers and UI caches must not reconstruct or mutate
  that state independently.
- Root `legacy/` and `todos/` are ignored local state. Versioned history belongs
  under `docs/legacy/`; shared plans belong under the owning `docs/**/plans/`.

Import and placement rules are executable contracts. Extend the focused
boundary test when changing one; do not duplicate the full package map here.

## Development ownership

Select the mode from local facts once. A worktree is Harness-managed only when
its enclosing generated workspace has a readable `task.env` and Task contract
whose Task ID and repository root match the current checkout. If those markers
are present but missing, contradictory, or stale, stop and repair the handoff;
do not silently fall back. With no managed markers, use standalone Git mode.

In Harness mode, its installed project guide, generated Task contract, scope,
writer lease, and evidence records govern managed work. Do not also run
`claim_worktree.py` or maintain a second branch-description record for that Task.
Standalone local development uses the lightweight Git workflow defined in
`docs/development/git-workflow.md` and `$integrate-shared-worktrees`; explicitly
identify the selected mode once, without repeatedly loading both workflows. It
requires no Harness installation or robot assets.
Coordinate with any existing owner before overlapping work, including managed
Tasks; switching mode is a handoff, never a way around an unresolved gate.

Inspect ownership and relevant peer scopes before writing and before handoff;
refresh when peers or the base change. Do not rescan unchanged historical
worktrees at every command. Peer worktrees remain read-only. Readiness binds
one clean commit and its evidence, not a moving branch name.

## Local resources

Run `python scripts/dev_local_resources.py audit` before handoff when ignored
resources or worktree-local links change. Only reviewed read-mostly assets may
use automatic cross-worktree links; mutable runtime or collection evidence
remains writer-isolated. Tracked code, tests, and profiles must not depend on a
developer-local absolute path.

## Safety

- Prefer offline and read-only checks. Do not start devices or issue motion,
  homing, replay, servo, force-control, reboot, reset, or other device-state
  changes without explicit scope for that action in the current request and
  the applicable safety preflight.
- Use the `codex_remote/` wrappers for remote hosts or hardware. Do not replace
  their fail-closed checks with ad hoc SSH.
- Before an overwrite or transform involving datasets, hardlinks, or media,
  prove source/target identity and an independent backup. Write fresh output,
  validate semantic counts and decodability, then replace atomically.
- Final composed control output, including persistent terms, requires an
  explicit bound at its authoritative boundary.
- After unexpected real motion, freeze the implicated Active path and preserve
  the exact command, configuration, logs, and process evidence. Diagnose
  read-only; do not tune or retry until the cause and missing safety gate are
  closed.

## Verification

Trace behavior, invariants, consumers and failures; review correctness, reuse,
and replacement or justified retention of old paths. Independently review
high-impact contract, safety, persistence, concurrency or instruction changes;
fix and recheck findings. Undecided semantics go to their owner; low-risk edits
need no extra reviewer.

Use `bash scripts/run_pytest.sh ...`; start with the default suite and selection
table in `tests/README.md`, loading later sections only for markers, costly
environments, or test migration. Maintain existing behavior tests before adding cases; preserve
required gates and deletion authorization. Run `git diff --check`.
Handoff covers behavior, consumers, findings/resolution, gaps and workspace
disposition under the selected workflow. Report offline, simulation, remote and
hardware evidence separately; none implies another.
