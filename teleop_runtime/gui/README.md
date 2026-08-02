# Compatibility GUI import paths

This directory no longer owns RealTeleop or UMI presentation/runtime logic.
Its modules preserve historic imports while the canonical owners are:

- browser transport, rebuildable UI state, and startup presentation:
  `control_station.web`;
- RealTeleop control availability and input precedence:
  `control_station.real_teleop`;
- UMI control availability and input mapping: `control_station.umi`;
- device/runtime status sampling and operator-triggered entity operations:
  `teleop_runtime.real_teleop`;
- dataset/task semantics and episode commitment: `collection_runtime` and the
  existing collection adapters.

Do not add a new implementation here. Compatibility modules must re-export or
alias one canonical owner and must not create queues, threads, devices, safety
state, or dataset state.
