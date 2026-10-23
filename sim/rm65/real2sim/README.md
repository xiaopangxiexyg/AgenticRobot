# Real2Sim fidelity boundary

This package is the simulation-owned boundary for making MuJoCo closer to the
real RM65 + DM-Tac system. It does not introduce a new runtime entrypoint and
does not change shared control, collection, dataset, or real-device semantics.

## What "close to real" means

A useful Real2Sim is not merely visually similar and is not merely able to
replay the same qpos. For the same timestamped action history, it should
reproduce the distributions that affect policy decisions:

1. the same action meaning, frames, TCP, limits, and command lifecycle;
2. the same clocks, delay, hold, interpolation, and sampling phase;
3. plant response including accepted command, servo target, measured state,
   saturation, and asymmetric left/right behavior;
4. camera geometry and timing before photometric randomization;
5. contact, gripper, object, and tactile outcomes relevant to task success;
6. the same causal observation cut and missing/stale-data behavior.

Every fidelity statement must therefore be tied to content-addressed evidence,
an explicit status, limitations, and held-out validation. A runnable scene is
not evidence that a component has been identified.

The offline DM-Tac free-motion fitter is
`python -m sim.rm65.real2sim.gripper_position_fit`. It consumes the standalone
hardware identification traces, fits left/right and close/open independently,
and writes a content-addressed position-only artifact. It deliberately rejects
contact-force claims when no external force gauge exists and never initializes
real hardware. Collection conditions and commands are documented in
`hardware/DMTAC_gripper/GRIPPER_ACTUATOR_IDENTIFICATION.md`.

## Current baseline (2026-07)

| Component | Current sim truth | Status | Main gap |
| --- | --- | --- | --- |
| Shared action/control semantics | Uses the shared qref/qcmd planning path | strongest existing layer | Keep as read-only dependency |
| Arm transport/plant | All-valid-data fit of per-joint delay, high-speed J5-J6 target lag, and MuJoCo position actuators | provisional | New untouched held-out validation, force limits, and contact/load dependence |
| DM-Tac free motion | Bilateral, direction-specific delayed rate/acceleration-limited path | identified for step response | Fast preemption remains provisional; contact/load response is unmodeled |
| Contact/grasp | MuJoCo contacts; some task paths still use debug contact latch | debug/provisional | Force/contact coupling, slip, compliance, and real success labels |
| Timing | Independent clocks exist in `TimingSpec` | provisional | Some JSON inputs inherit code defaults; real jitter/phase/drop distributions are absent |
| Camera geometry | Sim camera streams exist | provisional | Real intrinsics/extrinsics, 15 Hz behavior, rolling/exposure latency, masks |
| Photometry | Rendered RGB | unmodeled | Exposure, white balance, noise, blur, lighting/material distributions |
| Tactile | No policy-grade tactile observation model | unavailable | Geometry-to-DM-Tac signal and latency model |
| Formal state trace | Accepted/target/measured state are now separate | contract-complete | Add real/sim paired reports and thresholds |

These labels are deliberately conservative. Promote a row to `identified`
only through a versioned manifest and held-out evidence.

### Arm free-motion baseline: `20260721.v4`

The v4 profile uses fourteen valid paired records: ten clean captures and four
guardian-restored captures. Each pair contains the 120 Hz SDK-accepted qcmd
trace and independent approximately 200 Hz UDP measured-qpos feedback, with
zero reported command errors or disabled feedback rows. Guardian-restored
records retain weight 0.5 because they lack an in-band UDP stop row; clean
records use weight 1.0.

The four evening captures `221544/221548`, `222640/222650`, `224159/224204`,
and `224526/224530` were first evaluated without tuning against frozen v3.
Their position-error thresholds passed, but the maximum effective-delay
mismatch was 8 ms, above the inherited 5 ms gate. That failed v3 held-out
result is permanently recorded before the captures were explicitly converted
to v4 fit data. The nominal `220204/220211` traces were rejected because they
have no common command/feedback window; orphan feedback `223212` was also
rejected. Neither contributes samples or weight to the fit.

J1-J4 remain frozen because their millisecond-scale delay estimates were not
stable across clean sessions. Exact 1000 Hz MuJoCo replay selected a bilateral
J5-J6 update with a 50 degree/second velocity hinge and a 75 degree/second fit
cap. The exact residual delays, slopes, and caps are versioned in
`sim/rm65/actuation/profile.py` and the v4 manifest.

Across all fourteen fit sessions, v4 changes mean per-joint qpos RMSE from
0.360008 to 0.358448 degrees. The weighted objective improves 0.712%, the ten
clean sessions improve 1.244%, and the four newly converted clean sessions
improve 1.819%. However, the guardian-restored subset regresses 0.463%, the
worst-joint RMSE changes from 1.168741 to 1.185046 degrees, and maximum absolute
error changes from 9.177512 to 9.206431 degrees. V4 is therefore a useful
stage result but remains `provisional`, not an identification claim.

The exact v4 source index and fit manifest are
`artifacts/rm65_arm_all_valid_fit_20260721_v4_sources.json` and
`artifacts/rm65_arm_free_motion_20260721_v4.json`. V3 preserves its failed
held-out result, while v2 and v1 remain historical. Promotion to `identified`
requires a new clean capture collected after v4 is frozen and never used for
tuning.

The v4 fit manifest remains immutable when runtime implementation or
configuration changes. In particular, its source hashes describe the exact
core and MuJoCo adapter used for the 20260721 fit; tests compare those hashes to
fixed historical values rather than silently reinterpreting the fit against
the current files.

当前运行时源码绑定是
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261023_v10.json`。
它在冻结 v9 的基础上登记原生物体跟随、携物几何及物理步后保护：重新绑定
executor、position rig 与 planned TCP 三项既有源码，并显式登记七项相关依赖。
实现来源是 `7d66cb491d44c891ef41678f066b80d072f87315`；本次夹具与元数据
迁移不改变所绑定的运行时代码。所引用的 115 项消费者回归确实运行于该提交及
`c5b5fa16e7cdd202e30990e4b9c3c6dd086d9166` 源码树，未借用 v9 的旧报告。
首轮完整验收的 11 项夹具／绑定失败以及旧课程正向实验的会话构造拒绝均另行保留，
不将其记为通过；修复后的完整验收也单独记录。该绑定不提供完整抓放或真机验收，
未重拟合参数，`provisional` 与新 held-out `not_run` 保持不变。
v9 及全部更早清单按原字节保留。

v9 历史运行时源码绑定冻结于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261022_v9.json`。
它在冻结 v8 的基础上，登记固定左右单臂 Skill 会话的 planned-motion 更新，
并补充 `agent_skill_target.py` 依赖；会话只选一臂，另一臂保留实测碰撞支持，
默认仍为右臂。左臂抓取、夹爪、历史、空间推理及模型组合仍拒绝启用。
源码修订为 `e4ab4e5d0580dd4cae3d2ce8aa783979025e5c0c`，这是添加本清单之前
的实现提交；聚焦测试实际运行于 HEAD `11c34162e` 加 MERGE_HEAD `71faf8b1d`，
其暂存树与 e4 的 `84cc94c8` 完全相同，并非提交 e4 后重跑。
本清单、说明和回归迁移不改动所绑定的运行时代码。
v9 按精确哈希引用 78 模块、1893 通过／6 跳过的聚焦离线报告及其源码树绑定，
不把可选依赖跳过计为通过，也不把 v8 的 401 项报告解释成新实现的证据。
首轮完整验收发现旧绑定未更新，此问题和修复后的完整重跑另行留存于阶段验收记录。
本次没有重拟合、没有新增真机或 held-out 证据；`provisional` 与 held-out
`not_run` 保持不变。v8、v7、v5 和 v4 拟合清单均按原字节保留。

v8 历史运行时源码绑定冻结于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261018_v8.json`。
它绑定当时 actuator、MuJoCo executor、最终命令准入、rig／assembly，以及显式
目标接触的 oracle、projection、result、binding 和 planned-motion 源码，同时保留
原 simulation contract、相机登记、冻结 v4 拟合及 v7／v5 运行时绑定的精确身份。
本次相对 v7 改动两个既有源码文件，并新增五项接触路径源码依赖；没有调整执行器
参数、重新拟合或改写历史清单。14 模块的离线报告按内容哈希登记，包含原生接触、
撤销临时 guard 后的实测复核及只读候选预览；它不证明真机抓取、制动或 swept-TCP
安全。`provisional` 及 held-out `not_run` 保持不变，完整组合验收单独记录。

v7 仍冻结保留于
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20261017_v7.json`。
它记录当时最终 TCP command admission 的源码身份；其哈希不随新接触 guard 更新。

The v5 runtime-source binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260922_v5.json`.
It records the executor change that
excludes the expected B601 gripper self-contact pair from collision admission,
without rewriting the old fit evidence. This binding explicitly records no
arm-profile refit and carries forward the `provisional` claim and held-out
`not_run` status; the collision exclusion is a simulation runtime behavior
change, not new physical-response evidence.

The 20260829.v4 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260829_v4.json`.
It records the exact execution-continuity and lifecycle sources before the
B601 self-contact exclusion and retains the preceding 20260822.v3 binding.

The 20260822.v3 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260822_v3.json`.
It records the corrected left/right wrist-camera serial assignment, the 4B2 CAD
scene contract, and the exact archived 20260724.v1 registry. It is a historical
runtime contract rebind, not a refit or a new fidelity result.

The historical v1 binding remains immutable at
`artifacts/rm65_arm_free_motion_20260721_v4_runtime_binding_20260724_v1.json`.
Its exact camera-registry payload is retained at
`artifacts/rm65_camera_registry_runtime_20260724_v1.json`; consumers must not
compare that historical binding against the mutable current registry path.

### DM-Tac position baseline: `20260722.v1`

The bilateral suite `dmtac_bimanual_position_20260722_184856_698964` completed
all six runs with settings restored and no trace-queue drops. The fit uses 89
free-motion segments after two robust left-side RMSE outliers were removed;
44 separate held-out segments evaluate left/right close/open independently.
Held-out median normalized-position RMSE ranges from 0.0110 to 0.0189 and the
worst segment RMSE is 0.0313.

The frozen runtime artifact is
`artifacts/dmtac_gripper_position_20260722_v1.json`. It records the original
remote fit artifact and trace hashes, the 20 Hz / 0.002 software gate, all four
directional parameter sets, and the validation metrics. MuJoCo delegates the
normalized command path to `sim.rm65.actuation.DmtacPositionActuator`; only the
finger-servo/contact binding remains backend-specific.

The 0.5 s fast-preemption probe is retained as `evaluated_not_identified`:
endpoint median absolute error is small, but long reversing strokes reach
0.123--0.136 maximum normalized error under the zero-initial-rate model. Force,
contact compliance, friction, slip, and load dependence remain unmodeled because
the collection had no external force gauge.

## Package placement

RM65-specific fidelity code and evidence live at the RM65 embodiment level:

- `sim/rm65/actuation/profile.py`: versioned response parameters;
- `sim/rm65/actuation/core.py`: backend-independent accepted-qcmd, transport,
  raw-target, and filtered-target state machine;
- `sim/rm65/real2sim/`: evidence contracts, fit source indexes, and manifests;
- `sim/mujoco/actuation/executor.py`: thin MuJoCo actuator/measurement adapter.

The former `sim.real2sim` compatibility package has been removed. Import the
canonical `sim.rm65.real2sim` and `sim.rm65.actuation` packages directly.

## Sim-scoped execution plan

### P0: establish truth and auditability

- Use `Real2SimArtifactManifest` for every fidelity baseline. Bind the exact
  `SimulationContract`, task/profile inputs, source traces, calibration files,
  code revision, environment, resolved clocks, claims, and validations.
- Keep formal measurements layered as accepted qcmd -> actuator target ->
  measured qpos. Never compute tracking metrics from accepted qcmd alone.
- Freeze one deterministic probe suite under `sim/`: arm steps/chirps, gripper
  open/close, cube drop, contact onset, and camera calibration-board renders.
- Produce a machine-readable comparison report with per-side/per-joint metrics.

Exit criterion: a run can be reproduced from hashes, and every dashboard value
can be traced to either a real source artifact or a measured sim state.

### P1: identify the dynamics that dominate closed-loop error

- Fit left/right arm delay, gain/damping, force limits, rate limits, saturation,
  and noise from existing safe real traces. Prefer distributions over a single
  fixed delay when the data supports them.
- Fit grippers independently per side; separate free-motion response from
  contact response. Do not reuse the left trace as right-side evidence.
- Replace task-level contact latch results with physical contact/grasp metrics
  for formal evaluation. Retain latch only under an explicit `debug_only` mode.
- Identify object mass/inertia, friction, compliance, and reset distributions
  from measured task objects rather than reward tuning.
- Match the actual camera intrinsics/extrinsics, frame timestamps, sampling
  rate, latency, and dropout before tuning appearance randomization.

Exit criterion: held-out real traces meet declared thresholds for response,
contact, and camera geometry without fitting on the validation episode.

### P2: observation fidelity and robust policy training

- Add measured photometric distributions: lighting, exposure, white balance,
  depth holes/noise, motion blur, and occlusion.
- Add a DM-Tac observation adapter only after a defensible sensor model and
  time alignment exist; otherwise report tactile as unavailable.
- Derive domain-randomization ranges from residual real-vs-sim errors. Keep the
  identified nominal model fixed and version randomization separately.
- Run paired open-loop probes first, then closed-loop policy evaluation with
  identical action/observation contracts and causal cuts.

Exit criterion: robustness gains hold on held-out real episodes, and failures
can be assigned to a fidelity component rather than hidden by aggregate task
success.

## Minimum validation report

Each fit should report at least:

- timing: median/p95 command-to-motion delay, jitter, effective sample rate,
  dropout, and timestamp phase error;
- arm: per-joint RMSE/max error, settling time, overshoot, velocity/acceleration
  error, and saturation rate for left and right separately;
- gripper/contact: close-response error, contact-onset time, normal force/impulse
  proxy, slip distance, release behavior, and success confusion matrix;
- camera: reprojection error, pose error, frame-age distribution, crop/mask
  agreement, and depth validity statistics;
- task: state-transition and failure-mode confusion matrices in addition to
  reward/success rate.

Fit and held-out reports must reference different source artifact IDs. A failed
or not-yet-run validation remains part of the manifest; it must not be omitted.

## Artifact status rules

- `identified`: backed by referenced source artifacts; use only after a fit.
- `provisional`: physically plausible plumbing or partial evidence; limitations
  are mandatory.
- `debug_only`: useful for pipeline development but not a fidelity claim.
- `unmodeled`: the real phenomenon matters but the sim intentionally omits it.
- `unavailable`: the needed real observation/evidence is not currently present.

`Real2SimArtifactManifest.artifact_sha256` hashes canonical JSON excluding the
hash field itself. Loading a persisted manifest verifies that hash. Evidence
references must resolve to content-addressed `ArtifactRef` entries, while all
resolved timing rates are embedded so changes in code defaults cannot silently
change the interpretation of an old experiment.

## Ownership guardrails

RM65 profiles, reusable target dynamics, and evidence belong under `sim/rm65/`;
backend binding belongs under `sim/mujoco/`, and task selection belongs in sim
task configs. Tests may live under `tests/`, but should exercise only sim-owned
surfaces. Real logs and calibration outputs are read-only evidence. Changes to
`robot_motion/`, `teleop_runtime/`, `dataset_contracts/`, hardware behavior, or
canonical entrypoints require a separate mainline decision and are outside this
plan.
