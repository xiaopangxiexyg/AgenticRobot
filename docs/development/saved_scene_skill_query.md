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
The planner's ordinary obstacle clearance stays 20 mm. Since 2026-10-29 this
offline query defaults to the separately reviewed positive-distance start
egress mode: a measured start that is already closer than 20 mm but still at a
strictly positive distance may only move away, without new near pairs or hard
contact. Pass `enable_inherited_start_clearance=False` to `from_manifest`, or
`--strict-start-clearance` to the CLI, to reproduce strict start admission.
`--enable-inherited-start-clearance` remains accepted and selects the default.
The selected mode is included in the query identity and audit policy; strict
queries keep the identity they had under the former strict default. The keyword
does not come from an untrusted manifest or an individual model query. Neither
mode executes a trajectory or changes the live Gateway's safety policy; the
RealMan Gateway applies the same default to observed safe moves through its
own CLI (`--strict-start-clearance` opts out there as well).
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

## Explicit region queries

Inherited positive-start clearance is the default of this offline consumer;
select strict start admission explicitly with the `from_manifest` keyword or
CLI flag described above. A manifest field cannot change it. This consumer does
not change production planner or Gateway defaults.

`plan_relation_region(side, target_reference, offset_bounds_xyz_m,
offset_frame="reference", max_candidates=8)` uses the shared deterministic
region sampler already used by MuJoCo relation queries. The caller supplies
three closed offset intervals, not an implicit relaxation of a precise target.
Orientation is preserved. The method searches for one certified path within a
single 20-second budget, retaining q0 and the observed scene across candidates.
Each candidate gets at most 3 seconds, matching the existing MuJoCo search
policy. A per-candidate timeout leaves that candidate unresolved and permits
another candidate while shared resources remain. It does not prove infeasibility.
It checks the planned FK endpoint against the region; an ordinary pose solution
within arrival tolerance can still be rejected if it lies outside the region.
Use the original point query for point-arrival tolerance semantics.

Only ordinary endpoint/IK/path rejections permit another candidate. A start
state violation or unavailable native check stops the search. Timeout is
`query_incomplete`, distinct from `region_no_candidate_path`; eight sampled
failures do not prove that a continuous region is infeasible. Full attempt
records and FK region validation remain in `audit().region_queries`.

For fair atomic-versus-composed experiments, the application may set manifest
`query_budget` to `{"maximum_calls": 8, "maximum_planning_sec": 20.0,
"maximum_call_planning_sec": 3.0}`.
This same budget covers atomic and region calls over the session, including
candidate generation, final FK, validation and time spent between candidates.
The candidate deadline is rechecked before starting the lower planner; a
native call already in progress cannot be preempted. If a fatal native fault
arrives after its budget, the result remains incomplete and reports its
`terminal_fault_reason` without trying another candidate. Over-budget or
exceptional candidates cannot be exported. A completed point plan whose FK is
outside the requested region remains available only as a point-query diagnostic.
Model wait time is not charged.
Omitting it preserves the previous independent atomic-query behavior. Close
cannot race a region search. No region result is an executable motion grant.

The offline region query defaults to the `proximity` candidate policy. Set
manifest `region_candidate_policy` to `halton` to explicitly select the
original sampling order. The shared proximity sampler preserves orientation,
proposes the point nearest the saved TCP, then the region centre and face
centres. Proposal bounds are inset by the
existing arrival tolerance (narrow axes use their midpoint). This reserves room
for IK error without changing the requested region: final FK is still checked
against the original intervals. Candidate count and planning budgets do not
increase. This is an offline selection policy, not a new Agent tool or a change
to production MuJoCo/RealMan defaults.

## Tool-entity hover queries

A TCP offset above a surface point does not say where the gripper is. In the
saved RealMan poses the reviewed gripper box hangs 41–90 mm below the TCP and
its fingertip face centre lies 7–10 cm away from the TCP, so part of a TCP
region such as Z [0.08, 0.16] m can put the gripper into the target itself.

`plan_entity_hover_region(side, target_reference, tip_offset_bounds_xy_m,
clearance_bounds_m, orientation_policy="preserve", max_candidates=8)` uses
semantics `agenticrobot.relation.tool_entity_hover.v1` and places the gripper
box instead of the TCP:

- The entity is the simple gripper box used by grasp screening: the DMTAC
  GraspGenX alignment v5 palm and finger solids, including the fingertip
  collision extension, enveloped at the maximum opening (opening is ignored).
  It is bound only when the frozen tool profile names DMTAC M for that side and
  its tool TCP equals the reviewed 140 mm TCP; otherwise the query is rejected
  for that arm. This nominal profile match does not establish the installed
  hardware geometry or a physical calibration.
- `tip_offset_bounds_xy_m` bounds the horizontal offset, in reference axes, of
  the centre of the box's approach-side (fingertip) face from the anchor.
- `clearance_bounds_m` bounds the vertical gap between the box's lowest point
  and the support height: the highest observed voxel inside the box's
  horizontal footprint and at most 0.30 m above the anchor (never below the
  anchor itself), plus the cloud's voxel quantization margin. The same margin
  expands the footprint and height-window selection so a voxel overlapping an
  edge is not lost because its center lies outside. Higher points in the
  footprint are reported as overhead, not as support. Unobserved space is
  not assumed free; the planner still applies its own collision model.
- `orientation_policy` is explicit: `preserve` keeps the measured TCP
  orientation; `tool_axis_down` turns the TCP approach axis onto reference -Z
  with the least rotation.

Proposals are ordered nearest to the current gripper state, centre, face
centres, then Halton fill, inset by the arrival tolerance. They run through
the same budgeted region loop as `plan_relation_region` (20 s, 3 s per
candidate, optional session budget) and the same planner. A found plan is
accepted only if the achieved FK pose, re-measured with the same geometry,
satisfies the ORIGINAL bounds. The audit keeps per-candidate support height,
footprint and overhead counts, and the entity identity. This is an offline
query semantic, not a published Gateway Skill or a change to the existing TCP
relation and region semantics.
