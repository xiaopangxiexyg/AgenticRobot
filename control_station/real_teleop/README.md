# RealTeleop Control Station boundary

This package owns RealTeleop presentation policy, operator-intent mapping,
view models, and application workflow coordination. It must not open or poll
devices, sample Home, run calibration, own robot safety/control, or write an
episode. Those operations remain behind typed `teleop_runtime` and
`collection_runtime` ports.

During migration, the legacy RealTeleop entrypoint and compatibility module
may import this package. Boundary tests enumerate those exceptions exactly;
new reverse imports are forbidden.

The shared browser transport and rebuildable presentation cache live in
`control_station.web.calibration_web_ui`. RealTeleop-specific control
availability and input precedence remain in this feature package; they do not
poll the shared web queue themselves.

The optional progress feature follows the same boundary:

- `progress.py` owns typed Advance/Undo/Failure intents and rebuildable control
  availability;
- `progress_workflow.py` coordinates lower-owned ports through structural
  protocols only. It does not import `collection_runtime`, `teleop_runtime`,
  robot/session implementations, or a dataset writer;
- the workflow may request Hold/Pause after a durable Failure, but it writes no
  receipt itself until RealTeleop reports that the authoritative Session Hold
  and formal recorder Pause both took effect;
- an uninterrupted ordinary operator Hold may be promoted to Failure only when
  RealTeleop supplies its exact same-Episode Hold/Pause effects. The workflow
  preserves that original pause cut, records the later classification as
  availability time, and never manufactures or replays a second Hold;
- `control_station.web.real_teleop_progress.js` is an independent, optional
  presentation plugin over the existing RealTeleop page. It renders only
  rebuildable progress state and emits typed UI events; it owns no progress
  fact, Episode lifecycle, Hold, recorder, or robot operation;
- RealTeleop constructs progress semantics only with an explicit progress
  profile, and renders the plugin only with the separate
  `--collection-progress-gui` opt-in. Omitting the GUI flag preserves the
  existing page and A/B/C semantics; omitting both flags also restores the
  no-progress data path.

The first real-robot gate remains LINEAR-only. The selector is intentionally
capable of displaying explicit next-step choices, but RealTeleop must continue
to reject rooted-tree profiles until the branch workflow has its own gate.
