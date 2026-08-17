# Shared probes and maintenance tools

This directory contains versioned, non-canonical operator tools migrated from
the former Git-tracked `todos/probes/` directory.  Keeping executable tools
here prevents personal planning state from becoming a runtime dependency.

These scripts have mixed risk levels.  Some are read-only audits, while others
can train models, connect to services, transform datasets, or execute an
explicit `--apply` path.  A filename containing `audit`, `probe`, `verify`, or
`smoke` is not proof that a command is read-only.  Inspect the exact script and
its `--help`, identify its source and target paths, and follow the owning
dataset, deployment, simulation, or hardware safety rules before execution.

They are not canonical teleoperation, collection, replay, or deployment
entrypoints.  Those remain documented under `docs/` and in the repository root
`README.md`.

Experiment-specific learner, loss, training, and evaluation implementations
belong under `experiments/`. A launcher kept here for compatibility must stay
thin and delegate to that owner.
