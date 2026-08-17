# Policy-learning compatibility path

The canonical non-production learner and offline evaluation source now lives
under `experiments/policy_learning/`. This top-level package temporarily keeps
object-identical `policy_learning.arm_hold` imports available while concurrent
callers move to `experiments.policy_learning.arm_hold`.

Framework-specific integration remains under its owning deployment package,
for example `deployments/openpi/training/`. Runtime weight files and exported
policy artifacts are not Python source and belong under `model/` locally.
