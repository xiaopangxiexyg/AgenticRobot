# reBot B601-RS HITL source topology

Status: M0 source-boundary contract.  This document does not declare a runtime,
motion permit, hardware backend, simulator backend, or haptic-safety claim.

## Pinned upstream source

AgenticRobot records the official reBotArm Python project as an independently
versioned Git submodule:

```text
parent path:  hardware/reBotArm_control_py
canonical:    Seeed-Projects/reBotArm_control_py
portable URL: https://github.com/Seeed-Projects/reBotArm_control_py.git
commit:       5ba28acef46237eb6a7560658bbc43b06cf8a259
tree:         ddd34d2b400b3987d4875bd6a99bd29b74d39dac
```

The gitlink, not a floating branch, is the accepted source identity.  The
submodule has no nested gitlinks at this commit.  It supplies upstream source,
the B601-RS URDF and meshes, and reference examples.  It is not an
AgenticRobot Python package and is not installed implicitly with the parent.

## M0 non-effects

M0 deliberately adds no import, dependency, entrypoint, backend registration,
device discovery, CAN access, simulator startup, control profile, or default
behavior.  Existing teleop, replay, deployment, Control Station, Agent Gateway,
control-composition, simulation, and hardware entrypoints therefore retain
their prior import graph and runtime semantics.

The parent package discovery explicitly excludes the submodule namespace.  A
future integration must introduce a reviewed AgenticRobot-owned adapter rather
than importing vendor code from shared control packages or making upstream
examples into production entrypoints.

## Future ownership boundary

The planned implementation remains split by the existing repository rules:

- `hardware/rebot_b601_rs`: real CAN/MotorBridge adapter, preflight, final
  command limits, watchdog, and device lifecycle;
- `robot_motion`: backend-neutral model, spatial-wrench, haptic-command, and
  controller contracts, with no vendor-device initialization;
- `sim`: MuJoCo-only B601 backend and contact-wrench adapters, with no real
  hardware dependency;
- `teleop_runtime` and `control_station`: source binding and explicit,
  default-off application profiles;
- `control_composition`: only generic same-cut relative-residual semantics; no
  B601-specific branch;
- RPent: later semantic Skill/Program integration only, never the continuous
  residual or haptic servo owner.

Motion residuals and haptic feedback are separate channels.  The former enters
the existing single proposal/authority/safety/dispatch chain.  The latter maps
a bounded, timed follower wrench to a master haptic command and must not become
a second motion authority.

## Update and validation rules

An upstream update requires an explicit child commit review followed by a
parent gitlink update.  Do not add `branch = main`, a local filesystem URL, or a
generated copy of upstream source/assets.  Each update must report the child
commit/tree and the parent commit/tree/gitlink together.

Before any runtime integration, separately resolve and verify the Pinocchio and
MotorBridge dependency contract.  Before any real-device enable, add explicit
feedback validity, timestamp/sequence, command receipt, watchdog, saturation,
fault latch, and reviewed safe-stop behavior.  Simulation or offline checks do
not establish those hardware properties.
