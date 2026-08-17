# Experiments

`experiments/` is a containment boundary for versioned, reproducible, and
non-canonical hypothesis implementations. It is an offline application leaf,
not another production layer or a holding area for code with an unclear owner.

An experiment may own a baseline, loss, ablation, benchmark, training loop,
experiment-local configuration, and offline evaluation. It may consume stable
public contracts from production domains. Production packages must never
import `experiments`, and an experiment must not become the authoritative owner
of a shared schema, persistent data meaning, artifact integration, authority,
safety, dispatch, device/service lifecycle, or production entrypoint.

## Admission boundary

Code belongs here only when all of these are true:

- it tests a bounded algorithmic or empirical hypothesis;
- it is not required by a canonical collection, teleoperation, replay,
  deployment, simulation, GUI, or Gateway entrypoint;
- its input and output can be stated through existing contracts;
- deleting or promoting it does not break a production package;
- its validation is reported at the boundary actually exercised.

The label `experimental` does not relax safety rules or authorize hardware,
remote services, dataset mutation, interactive simulation, or real motion.
Motion-capable lifecycle, canonical safety, and dispatch remain in their owning
runtime/profile even when an experiment supplies a candidate algorithm or
configuration.

## Related directories

| Content | Owner |
|---|---|
| reusable dataset projection, windows, sampling, and batch views | `dataset_interface/` |
| stable schemas, identities, and hash domains | `dataset_contracts/` |
| model clients, protocols, and artifact integration | `deployments/` |
| framework-specific training integration | `deployments/<framework>/training/` |
| checkpoints, weights, and runtime sidecars | local ignored `model/` paths |
| thin audits, maintenance commands, and compatibility launchers | `scripts/probes/` |
| behavior and boundary verification | `tests/` |
| obsolete implementation with no active caller | `legacy/`, after legacy-policy checks |

Experiment-specific commands may live beside their implementation. A file in
`scripts/probes/` should remain a thin launcher or cross-domain diagnostic; it
must not regain the learner, loss, or evaluation implementation.

## Dependency direction

Experiments may depend inward on explicitly selected public domain APIs. The
reverse direction is forbidden. Different experiment families keep their own
dependency allowlist; the existence of one simulation experiment, for example,
does not authorize a policy-learning experiment to import `sim`.

Do not create catch-all `common`, `misc`, `utils`, or `helpers` packages here.
When two experiments need the same stable semantics, first identify the real
production owner and promote the reusable code there.

## Source and outputs

Keep source and small reproducibility configuration under the experiment.
Write checkpoints and weights under ignored `model/<experiment>/<run-id>/`
paths, local run metrics under
`runtime_artifacts/runs/experiments/<experiment>/<run-id>/`, and datasets under
their dataset storage and manifest rules. Do not write generated artifacts,
caches, logs, videos, datasets, or secrets into `experiments/`.

Importing an experiment must not initialize devices, services, browsers,
viewers, workers, or filesystem outputs. All resource use and writes require an
explicit command and remain subject to the owning domain's safety rules.

## Lifecycle

An experiment can be `incubating`, `validated-offline`, `promoted`, or
`retired`. `validated-offline` is not production, simulator, or hardware
acceptance. Promotion means moving reusable logic into its production owner;
production code must not import the old experiment path. Retired code is
removed or archived only after callers and replacements have been checked.

Each experiment family keeps a concise README stating its purpose, non-goals,
status, inputs, outputs, commands, dependency boundary, validation, and next
possible production owner. No separate registry is required.
