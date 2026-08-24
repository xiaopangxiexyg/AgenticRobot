# Robosuite environment boundary (A0)

This directory reserves the project-facing configuration boundary for future
Robosuite environments. A0 deliberately contains no concrete environment,
asset, installation hook, or runtime launcher.

The implemented seam is under `sim/robosuite/`; its backend is injected and
the A0 acceptance path uses only the deterministic offline mock. Binding a real
Robosuite environment requires a later task with an explicit environment
contract, dependency/runtime validation, and simulator authorization.
