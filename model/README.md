# Local policy model artifacts

`model/` is reserved for local policy weights, checkpoints, and their runtime
sidecars. It is intentionally not a Python package and must not contain learner,
loss, evaluation, robot-motion, or deployment source code.

The directory contents are ignored by Git except for this README. Versioned
artifact contracts and integration code belong under `deployments/`; reusable
learner source belongs under `policy_learning/`.
