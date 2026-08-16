# Local policy model artifacts

`model/` is reserved for local policy weights, checkpoints, runtime sidecars,
and explicitly pinned external source boundaries. It is intentionally not a
Python package and must not contain AgenticRobot-owned learner, loss,
evaluation, robot-motion, or deployment source code.

The directory contents are ignored by Git except for this README and the
tracked `model/openpi` Git submodule. OpenPI remains independently owned and
uses its own Python/runtime environment; its gitlink pins only the external
source version. Versioned artifact contracts and integration code belong under
`deployments/`; reusable learner source belongs under `policy_learning/`.
