# Policy learning

`policy_learning/` owns framework-local learner definitions, losses, training
batch adapters, and offline model evaluation. It may consume
`dataset_contracts` and `dataset_interface`, but it does not own dataset
schemas, robot-motion semantics, policy deployment, safety, authority, or
dispatch.

Framework-specific integration remains under its owning deployment package,
for example `deployments/openpi/training/`. Runtime weight files and exported
policy artifacts are not Python source and belong under `model/` locally.
