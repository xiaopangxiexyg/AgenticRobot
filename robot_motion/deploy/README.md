# robot_motion.deploy

`robot_motion.deploy` holds sim/real-neutral runtime glue for motion-control
deployment. Model-server client code lives under `deployments.client`.
Real-only hardware adapters stay under `teleop_runtime`; sim-only adapters stay
under `sim`.

Current boundary:

```text
ObservationSource -> Policy -> CanonicalActionPipeline -> Dispatcher
```

- `source.py`: common observation-source protocol.
- `dispatcher.py`: common command-dispatch protocol and dry-run recorder.
- `runtime.py`: small common single-step runtime skeleton.

Migration notes:

- Large action/deploy types still live in `teleop_runtime.deploy` during this
  transition. Do not move them in one large edit.
- New motion-control common code should be added here first.
- Generic model-server client code belongs in `deployments.client`.
- Model-family-specific client code belongs in `deployments/<model>/client`.
- Real launchers may import from this package; this package should not import
  real hardware backends.
- Sim launchers may import from this package; this package should not import
  MuJoCo-specific backends.
- Cross-domain launchers such as real-observation-to-sim-dispatch should live
  under `deployments/dexteleop` or another composition layer, not in
  `robot_motion`.
