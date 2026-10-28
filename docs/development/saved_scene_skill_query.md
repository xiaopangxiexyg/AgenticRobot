# Saved-scene relation planning

`scripts.probes.query_safe_move_snapshot.SavedSceneSkillQuery` is an offline
consumer of the production surface selection, spatial-goal solver and observed
safe TCP planner. It supports direct lower queries and Agent experiments on the
same saved time node. It does not advertise a new production Skill or replace
the admitted RealMan/MuJoCo Gateway execution lifecycle.

First export an accepted operator sample through the existing
`load_operator_scene_sample` entry when that separate codec leaf is installed.
The resulting `agenticrobot.safe_move_inputs.v1` directory, frozen context and
matching URDF are the input. An incomplete sample must not be repaired by
inventing timestamps, qpos, camera poses or missing views.

The manifest schema is `agenticrobot.saved_scene_query.v1`. It contains absolute
`replay_dir`, `context_dir`, `urdf`, explicit boolean `source_single_start`,
optional `filter_config`, and `fixed_files`
(absolute file path to SHA-256). Freeze snapshot JSON, every RGB-D NPZ, matching
URDF, all seven calibration/configuration inputs and optional filter config.
Each optional `anchors` entry contains `label`, `role`, `mask_path`,
`mask_sha256`, `rgb_content_hash`, `depth_content_hash`, and `acquired_at`.
The saved mask must belong to that exact RGB-D frame. Labels describe model
evidence; they are not operator-provided object truth. Existing default self
filtering remains unchanged; expanded filtering requires an explicit config.
The planner also keeps the original strict 20 mm start-clearance default.
To inspect the separately reviewed positive-distance egress mode, explicitly
pass `enable_inherited_start_clearance=True` to `from_manifest`, or add
`--enable-inherited-start-clearance` to the CLI. The mode is included in the
query identity and audit policy; omitting it and passing False are equivalent.
The keyword does not come from an untrusted manifest or individual model query.
The original source branch's implicit egress results remain historical evidence;
they are not results for this integrated strict default. Neither mode executes
a trajectory or changes the live Gateway's safety policy.
Operator exports additionally require `source_record`: freeze the original
record, observation ZIP and all its context files. The probe verifies the
exported arrays/geometry against that original archive and binds calibration to
the record. The declared source startup policy must reconstruct its exact camera
binding; a mismatch is rejected, never retried with another policy. Saved
observation depth/voxel settings are preserved. For original safe-move records,
the request's q0 takes precedence over exposure-time qpos. An operator export
without a planning request uses its qualified stationary qpos association.

```python
import json
from pathlib import Path
from scripts.probes.query_safe_move_snapshot import SavedSceneSkillQuery

query = SavedSceneSkillQuery.from_manifest(json.loads(Path(manifest_path).read_text()))
try:
    scene = query.read_scene()
    # Select an actually returned reference according to the task.
    ref = scene["references"][2]["reference"]
    result = query.plan_relation(side="right", target_reference=ref,
                                 offset_xyz_m=[0.0, 0.0, 0.08])
    # Preserve query.audit() separately; export a path only when path_found.
finally:
    query.close()
```

`read_scene()` returns short references, semantic labels, measurement times and
the actual planning frame. Coordinates, calibration hashes and native details
remain in `audit()`. For an atomic-coordinate baseline only,
`read_scene(include_geometry=True)` and `plan_pose()` use the same observation,
safe planner and limits. Relation offsets use the returned reference frame or
the original measured TCP axes. Orientation is preserved. The model never
supplies a source hash or changes the recorded robot state.

Each planning call starts from the original measured qpos and uses the same
observed obstacle cloud. Reusing an anchor reuses evidence; it does not track a
moving object or certify its present position. Multiple calls are independent
alternatives, not a simulated execution sequence. `path_found` is a full
planning result, not arrival, grasp, release, or task success. Native faults
propagate separately from ordinary planning rejection. Close cannot report
completion while a native query is running.

For inspection or a single JSON relation request:

```bash
python -m scripts.probes.query_safe_move_snapshot --manifest /absolute/query.json
python -m scripts.probes.query_safe_move_snapshot --manifest /absolute/query.json \
  --request /absolute/relation.json --audit-output /absolute/new-audit.json
```

The caller owns the existing native-resource lock and fixed experiment
denominator. This probe has no transport, camera/backend startup, dispatch,
domain-HOLD, or executable-plan retention. Actual RealMan motion remains a
separate, freshly observed and authorized Gateway request.
