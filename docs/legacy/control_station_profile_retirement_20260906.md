# Control Station CS-2/CS-3/CS-6 profile retirement

Date: 2026-09-06 (Asia/Shanghai)
Task: `AR-20260906-control-station-profile-retirement-r0`
Source checkpoint: `cbb0b2c9b52a87bf7d4b15be444be244cda69054`

## Decision

The following application-leaf profile modules had no production caller,
runtime entrypoint, or compatibility import outside their own fixture tests:

| Retired module | Profile family | Caller audit and replacement owner |
|---|---|---|
| `control_station/composition/mujoco_human_station.py` | CS-2a2b scoped synthetic-human Station | Only CS-2a2b fixture tests called the builder. Keep the shared CS-2a1/2a2a human binding in `mujoco_human.py`; use the current typed Station/target seams for any future application binding. |
| `control_station/composition/mujoco_replay.py` | CS-2b/2bc replay-to-MuJoCo fixtures | Calls were limited to replay fixture tests. Canonical replay emissions and cursor ownership remain in `teleop_runtime.replay` / `robot_motion.replay`. |
| `control_station/composition/mujoco_policy.py` | CS-2c synchronous policy fixture | Calls were limited to its fixture tests. `PolicyActionEmission` and policy runtime ownership remain in `robot_motion.deploy` / `teleop_runtime.deploy`; current Sim-HIL uses the component profile. |
| `control_station/composition/mujoco_rtc_policy.py` | CS-2d RTC-to-MuJoCo fixture | Calls were limited to its fixture tests. RTC frontend/protocol ownership remains in `teleop_runtime.deploy.rtc_frontend` and `robot_motion.deploy`; source preflight remains in `control_station/preflight/openpi_rtc.py`. |
| `control_station/composition/rtc_policy_shadow.py` | CS-3 passive RTC Station shadow | Its Station bootstrap was only exercised by fixture tests. Passive emission observation uses the lower `rtc_step_fanout` seam and the pure preflight projection; no Station lifecycle or producer owner is recreated. |
| `control_station/composition/mujoco_human_replay.py` | CS-6a/6b0/6b1d/6b1f replay-human handoff fixtures | Calls were limited to CS-6 fixture tests and historical guide text. Current passive source, HITL lifecycle, and component composition owners are `teleop_runtime.action`, `control_station/hitl`, `control_composition`, and the default-off `control_station sim-hil` profile. |
| `control_station/composition/mujoco_pose_composer.py` | CS-6b1h/6b1i pose-composer/session fixtures | Calls were limited to composer/session fixture tests. Current pure strategies and allocation/session ports remain in `control_composition` and the active Sim-HIL profile. |

The five CS-2/CS-3 families and the two CS-6 modules were parallel adapters
around existing lower owners. Keeping them would preserve duplicate Station
adapters, publication barriers, and lifecycle paths without a live caller. The
retirement removes those adapters and their dedicated tests; it does not remove
the lower-owned replay, policy, RTC, human, safety, IK, dispatch, or capture
implementations.

The generic Web functions
`build_scoped_sim_authority_control_station_server` and
`ScopedSimAuthorityAccessPolicy` remain for now. They have no in-repository
caller after this change, but they are a public Web contract outside this
Task's write scope and require a separate owner/compatibility decision before
removal. Likewise, CS2A2B manifest constants shared by the retained human
binding are not deleted from `mujoco_human.py` in this task.

## Versioned change

Removed from the versioned source tree:

- seven profile modules, 8,578 production lines;
- their dedicated fixture tests and the CS2A2B Web authority fixture;
- CS2A2B/CS-3-only portions of the capture and RTC frontend tests;
- stale current README/Sim-HIL runnable references and the obsolete boundary
  entries.

The remaining generic Web, lower deploy/RTC, and active Sim-HIL tests continue
to cover the reusable contracts. Historical architecture reports and generated
evidence are intentionally unchanged; they are not current runnable commands.

## Recoverable local backup

Before deletion, every removed source/test and the locally edited source files
were copied to:

`/home/zc/HandMotion/Developing/HarnessLocal/agentic-robot/Worktrees/AR-20260906-control-station-profile-retirement-r0/AgenticRobot/legacy/control_station_profile_retirement_20260906/source/`

The ignored backup contains `manifest.tsv` (sizes), `manifest.sha256` (SHA-256
values), and `README.md`. Verification performed before checkpoint:

```text
sha256sum -c legacy/control_station_profile_retirement_20260906/manifest.sha256
```

The command passed. The root `legacy/` directory is Git-ignored and is not a
Python import path; no backup file is tracked or referenced by production code.
Restore requires a new ownership, compatibility, and safety review.

## Safety and validation boundary

This retirement was source-only. No simulator, camera, VIVE, Quest, RM65,
B601, hardware backend, homing, replay motion, force/impedance command, or
real action was started. No hardware configuration, `main`, remote, or peer
worktree was changed, and nothing was pushed. Offline tests only establish
source/API behavior; they do not establish physical safety, calibration,
obstacle clearance, real-time performance, or deployment readiness.

Before commit, the owner must re-run the full-repository reference scan,
`git diff --check`, backup hash verification, Harness conflicts, and the
required AgenticRobot Profiles. A changed tip or new semantic overlap creates a
new candidate and blocks handoff until re-reviewed.
