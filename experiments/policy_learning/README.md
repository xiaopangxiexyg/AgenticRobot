# Policy-learning experiments

This subtree owns non-canonical learner definitions, losses, training commands,
baselines, and offline evaluation used to test policy-learning hypotheses.
Current status is `validated-offline`; that status does not establish a
production policy source, deployment path, simulator acceptance, or motion
readiness.

Policy-learning experiments may import `dataset_contracts` and
`dataset_interface` plus scientific Python dependencies. They must not import
`collection_runtime`, `control_composition`, `control_station`, `deployments`,
`hardware`, `robot_motion`, `sim`, or `teleop_runtime`. Those packages must not
import this subtree.

Data windows and deterministic dataset sampling remain in `dataset_interface`.
Framework-specific preparation and launch integration remain in
`deployments/<framework>/training`. Weights and checkpoints remain in ignored
local `model/` paths.

`arm_hold/` contains the current lightweight learner and its offline evaluation
commands. It is a bounded experiment family, not the seed of a general training
framework. If a reusable learner platform is later justified by multiple real
consumers, promote it to an independently reviewed owner instead of widening
this experiment boundary.
