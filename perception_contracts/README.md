# Perception artifact contract boundary

`perception_contracts` owns runtime-independent identities for immutable
perception artifact bytes: RGB images, metre-valued depth arrays, localization
input and annotated images, and point-grasp inputs and visualizations.

It is shared by perception Skill bindings, camera/simulator adapters, and
tests.  It defines neither dataset layout nor collection/replay semantics and
must not import runtime, hardware, simulator, model, or dataset packages.

The dependency direction is intentionally one-way:

```text
agent_gateway / sim / deployments  --->  perception_contracts
```
