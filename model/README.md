# Local policy model artifacts

`model/` is reserved for local policy weights, checkpoints, runtime sidecars,
and explicitly pinned external source boundaries. It is intentionally not a
Python package and must not contain AgenticRobot-owned learner, loss,
evaluation, robot-motion, or deployment source code.

The directory contents are ignored by Git except for this README and reviewed
Git submodule boundaries. `model/openpi` remains independently owned and uses
its own Python/runtime environment. `model/perception/{sam2,GroundingDINO,vgn,
graspnet-baseline,graspgenx}` supplies the pinned external source used by the
optional perception installers. Each gitlink records only an upstream commit;
model weights and GraspGenX's isolated service environment remain in the
reviewed `assets/` store.

Versioned artifact contracts and integration code belong under `deployments/`;
non-canonical learner experiments belong under `experiments/policy_learning/`.
Initialize the external sources after clone or pull with:

```bash
git submodule update --init --recursive -- \
  model/openpi model/perception/sam2 model/perception/GroundingDINO \
  model/perception/vgn model/perception/graspnet-baseline \
  model/perception/graspgenx
```
