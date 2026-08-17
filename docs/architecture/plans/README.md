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

- [Control-core landing](dexteleop_control_core_landing_plan.md): current direct
  versus composed execution boundary, shared-core ownership, maturity terms,
  and remaining structure/authority gates. The original M0 to M8 batch ledger
  is preserved under [`../reports/`](../reports/control_core_landing_ledger_20260722.md).
- [Control Station](dexteleop_control_station_plan.md): current application
  control-plane ownership, strict GUI/browser subset, source/target profile
  debt, and remaining boundary migrations. Its completed CS-0 to CS-6 ledger
  is preserved under [`../reports/`](../reports/control_station_implementation_ledger_20260724.md).
- [RTC landing](dexteleop_rtc_landing_plan.md): current raw/model RTC producer
  ownership, timing/representation invariants, default-off profiles, re-arm,
  transport, and downstream authority gates. The original RTC-0 to RTC-7 and
  R-R/MRTC ledger is preserved under
  [`../reports/`](../reports/rtc_landing_ledger_20260725.md).
- [HITL and chunk-relative control](hitl_chunk_relative_migration_plan.md):
  current exclusive/continuous Sim-HIL profiles, application/composition/
  trajectory ownership, strict GUI subset, real-binding limits, and remaining
  handback gates. The original R0 to R4 and R3.1a-c ledger is preserved under
  [`../reports/`](../reports/hitl_chunk_relative_ledger_20260807.md).

## Structure-remediation ledger

- [P0 structure remediation](dexteleop_p0_structure_remediation_plan_20260803.md):
  current gates for the remaining UMI compatibility audit, RealTeleop resource-
  lifecycle decomposition, and RM65 solver work. The 2026-08-04 to 2026-08-07
  implementation ledger is preserved under
  [`../reports/`](../reports/p0_structure_remediation_20260803.md); neither file
  replaces a fresh audit of the branch that will receive a new batch.
