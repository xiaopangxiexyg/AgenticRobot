# Shared architecture plans

This directory contains cross-module plans that are referenced by current
architecture, schema, or legacy-navigation documents and therefore need to be
available in every clone.

Plans describe incomplete work and do not override current code, tests,
`AGENTS.md`, or public schemas. A completed sub-batch recorded in a landing
ledger does not make the whole plan complete; use the plan's current-status
section and remaining gates rather than its filename or modification time.
Personal notes and transient task lists belong in the Git-ignored root
`todos/` directory; reusable plans enter this directory only when a tracked
consumer needs them.

## Long-term architecture and observability

- [Target architecture](dexteleop_architecture_planning.md): cross-domain
  semantics, ownership invariants, and the control/data/training loop.
- [Control composition observability](dexteleop_control_composition_observability_plan.md):
  observability requirements and staged adoption for composed control.

## Active landing ledgers

- [Control-core landing](dexteleop_control_core_landing_plan.md): shared
  control-core batches, maturity levels, authority gates, and validation ledger.
- [Control Station](dexteleop_control_station_plan.md): application control
  plane, source/target bindings, and remaining station integration batches.
- [RTC landing](dexteleop_rtc_landing_plan.md): action-chunk producer,
  transport, admission, target handoff, and real-observation gates.
- [HITL chunk-relative migration](hitl_chunk_relative_migration_plan.md):
  relative HITL session, handback, coordinator, device binding, and durable
  evidence boundaries.

## Structure-remediation ledger

- [P0 structure remediation](dexteleop_p0_structure_remediation_plan_20260803.md):
  completed structural slices plus the still-applicable follow-up conditions.
  It remains here while those conditions are referenced; it is not a current
  worktree snapshot and must be revalidated against the present branch before
  starting a new batch.
