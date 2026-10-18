# Local policy model artifacts

`model/` is reserved for local policy weights, checkpoints, runtime sidecars,
and explicitly pinned external source boundaries. It is intentionally not a
Python package and must not contain AgenticRobot-owned learner, loss,
evaluation, robot-motion, or deployment source code.

The directory contents are ignored by Git except for this README and reviewed
Git submodule boundaries. `model/openpi` remains independently owned and uses
its own Python/runtime environment. `model/perception/{sam2,GroundingDINO,vgn,
graspnet-baseline,graspgenx}` supplies the pinned external source used by the
optional perception installers. Each gitlink records an exact source commit.
GraspGenX uses the `CYBER-QAQ/GraspGenX` fork to retain the reviewed
support-aware side-grasp changes; publish that pinned commit in the fork before
publishing a parent revision that requires it. Model weights and GraspGenX's
isolated service environment remain in the reviewed `assets/` store.

Versioned artifact contracts and integration code belong under `deployments/`;
non-canonical learner experiments belong under `experiments/policy_learning/`.
Initialize the external sources after clone or pull with:

```bash
git submodule sync -- model/perception/graspgenx
git submodule update --init --recursive -- \
  model/openpi model/perception/sam2 model/perception/GroundingDINO \
  model/perception/vgn model/perception/graspnet-baseline \
  model/perception/graspgenx
```
