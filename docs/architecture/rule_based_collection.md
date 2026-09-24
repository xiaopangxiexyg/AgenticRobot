# Rule-based collection boundary

## 正式书写的 Progress 完成语义

规则执行器通过靠近、书写、离开三阶段的已接纳样本覆盖检查后，才向正式 Episode
所有者返回 `rule_based_complete`。Progress adapter 还要求有效的三阶段绑定、活动
Attempt，以及当前步骤仍为 `right_write`，然后按既有录制范围完成，而不是将规则
书写一律当作截取窗口：

- `FULL_TASK` 还要求当前步骤为任务终点，封存结果为 `FULL_TASK_COMPLETED`、
  Attempt `SUCCESS`、`full_task_outcome=SUCCESS`。
- `EVENT_WINDOW` 保持 `WINDOW_COMPLETED`、Attempt `SUCCESS`、
  `full_task_outcome=NOT_APPLICABLE`；不能把局部窗口声明为完整任务成功。

中途 Hold、取消、急停、故障或无效完成绑定不走该成功分支。样本覆盖检查、正常
writer 收尾和 Progress 封存仍由原有实现负责。本判断只修复后续 Episode 的收尾
语义，不修改已封存历史数据、绑定或 manifest；历史修正需要另行授权的数据修复流程。

## 规则采集随机种子

`scripts/run_rule_based_writing_collection.sh` 默认在每次启动时从操作系统随机源生成一次种子，随后将同一个整数交给来源池选择与字形、方向、大小选择。单次运行内的轨迹序号仍决定逐条选择；重启后得到新的选择序列。启动日志以及自动来源、随机字形选择的元数据记录实际种子，便于追溯。

`--selection-seed auto` 显式使用这一默认行为。传入 `--selection-seed 42` 等整数会有意复现顺序；复现还要求来源池内容、候选范围、字形目录和随机选项一致。旧启动命令若仍带固定种子，重启时会继续重复同一顺序。

## 可回退的书写接触点校正

书写专用启动脚本可选 `--correct-tcp`。默认不启用，普通遥操和原有规则采集仍使用右臂 DMTAC_M 配置的法兰局部 +Z 0.14 m `tool_tcp`。启用后，高度标定以法兰局部 +Z 0.20 m 的实际夹具末端为参考；书写、落笔下降、靠近/离开阶段的高度下限和对应可达性预检，都先按当前姿态把夹具末端高度换算成原有 `tool_tcp` 的目标。命令的受控坐标系仍是 `tool_tcp`，继续由原有 IK、碰撞、位置与受力安全链路处理。数据中的高度标定元数据会标出夹具末端参考系与 0.06 m 偏移。

该开关要求右臂实测状态报告的工具 TCP 仍为法兰局部 `[0, 0, 0.14, 0, 0, 0]`；不符或缺失时标定拒绝继续。移除开关并重新采集高度即可回到原行为，不应把两种几何定义的数据混入同一任务名。离线检查不证明新增夹具的碰撞几何已经被现有代理模型覆盖；现场运动前仍须确认夹具与桌面、相机杆及双臂的间隙，并由操作员按安全流程验收。

The rule-based collector is a small plan-composition seam for collection
workflows that need a deterministic combination of existing operations, for
example:

```text
initialization -> replay frames 10..24 -> custom action x3 -> replay frames 80..90
```

Its implementation is `robot_motion.replay.rule_based`. The collection-facing
bridge is `teleop_runtime.collection.rule_based_adapter`; it owns only:

- ordered declarative steps (`RuleBasedInitialize`, `RuleBasedReplay`, and
  `RuleBasedCustomAction` / `RuleBasedActionSegment`);
- expansion of those steps into `RuleBasedCollectionDirective` values; and
- per-step `RuleBasedCaptureSelection` intent; and
- resolution of each directive into the existing `RobotAction` /
  `TeleopCommand` boundary for a caller that already owns a collection loop.

It deliberately reuses the existing `ReplayTrajectoryGroup`, qpos-init
selection, replay action construction, and canonical `RobotAction` APIs.

## Explicit non-ownership

The rule-based layer does not:

- call hardware, start a device, dispatch a robot command, solve IK, or apply
  safety/admission;
- own a collection loop, worker, queue, episode lifecycle, or dataset writer;
- define or replace a dataset schema; or
- create a second replay implementation.

`RuleBasedCollectionAdapter` consumes each directive and resolves its action;
the caller then applies the resulting command through the normal lower runtime
and submits a sample only when `emission.recordable` is true. An empty
`stream_ids` allowlist means all already-bound streams, while a populated
allowlist exposes the requested stream selection to the caller. The adapter
does not rewrite samples or writer jobs, so the existing sample builder and
writer remain authoritative for causal cuts, validation, persistence, sealing,
and commit.

## Runtime rule-based capture segments

`RuleBasedCustomAction` is appropriate for a fixed canonical action repeated a
known number of ticks. For a rule-based segment whose action depends on the
current runtime observation, use `RuleBasedActionSegment` instead. It reuses
the existing `RobotActionSource.produce_action(observation, context=...)`
contract and adds only two collection-plan facts:

- `RuleBasedCaptureSelection` says which already-bound streams are eligible for
  this segment;
- `RuleBasedSegmentStopCondition` supplies a mandatory hard `max_steps`, an
  optional `min_steps`, and an optional `stop_when` predicate evaluated after
  each produced action.

The runtime caller advances such a segment with
`RuleBasedCollectionCursor.next_runtime(observation, context=...)`. A source
returning `None` is rejected rather than interpreted as an implicit stop. The
terminal directive is returned with `segment_terminal=True` and an explicit
`segment_stop_reason`; only the next cursor request advances to the following
plan step. This gives the caller a precise point to finish the capture, read
the actual TCP state, publish a `RuleBasedBoundaryState`, and then request the
next relative replay.

The action source's `start()`/`close()` lifecycle is intentionally not called
by the plan cursor. The existing runtime that owns the source remains
responsible for lifecycle, observation acquisition, dispatch, safety, and
writer interaction.

## Relative replay at a segment boundary

The pure boundary contract is `RuleBasedBoundaryState` together with
`rebase_replay_action_at_boundary`. It uses the source trajectory's recorded
actual TCP at `source_start_index` as the source anchor, converts each source
pose target into a fixed relative delta, and applies that delta to the
runtime-measured TCP in `RuleBasedBoundaryState`. The result is an absolute
`RobotAction` in the already-validated per-side pose frames.

This first seam is intentionally pose-only and fails closed when the source
actual TCP history or per-side frame identity is missing. Joint-qpos replay is
not treated as a relative pose replay. A runtime action segment publishes its
terminal directive before the cursor resolves the following step, so the
boundary can be measured from the actual post-segment state rather than from a
planned action target.

This module is a pure, default-off composition seam. Its unit tests use only
synthetic trajectory arrays, canonical action objects, and an in-process
source; they are not evidence of robot readiness or production collection
integration. No existing CLI or GUI entrypoint is changed by this adapter;
adoption is an explicit caller choice. The existing collection adapter still
owns only command resolution; production runtime wiring for
`next_runtime(...)`, boundary measurement, and dataset submission remains a
separate integration step.

## MuJoCo composition proof

The simulation-only example
`sim.mujoco.replay.rule_based_multistage_demo` exercises the same boundary
without adding a second control or writer implementation. Its sequence is:

```text
initial alignment (not recorded)
  -> absolute replay of --stage-a
  -> right TCP fixed trajectory in desk XY at constant desk Z
  -> measured actual-TCP boundary
  -> relative replay of --stage-b
```

Run it with an external canonical UMI dataset root:

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m sim.mujoco.replay.rule_based_multistage_demo \
  --dataset-root /path/to/dataset \
  --episode 0 --stage-a 0:120 --stage-b 250:400 \
  --output /tmp/rule_based_mujoco_demo --record-depth --no-sleep
```

`start:end` spans are half-open and intentionally editable. The output keeps
the normal simulation parquet/video layout and adds
`rule_based_composition_manifest.json`; each row carries its phase in
`sample_metadata_json`. For this composition demo the writer also carries the
source canonical pose-frame contract and writes the action target in
desk-absolute xyz-quat form, so the result can be checked with the existing
UMI loader rather than being only a visualization artifact. The relative phase
fails closed if the source group does not contain actual TCP history or if the
source/boundary pose frames do not match. The optional passive viewer is
MuJoCo-only (`--viewer`); rendered RGB/depth videos are also produced for
headless verification.

The rule-based circle defaults to the selected source spans' right-TCP average
linear speed in the canonical desk frame. Use `--circle-speed-m-s` for an
explicit target speed; `--circle-hz` is only an explicit fixed-waypoint-rate
compatibility override. The demo requires the source dataset contract hash to
match the MuJoCo station pose contract, and records the calibrated
`world_from_desk` mapping in `pose_frame_alignment`; it does not silently apply
an additional raw-to-desk transform.

The intermediate phase can instead use a checked-in one-stroke digit
centerline. Its authored height is adjustable with
`--fixed-trajectory-size-m` (the compatibility alias
`--fixed-trajectory-scale-m` is also accepted), and its orientation is one of
four named directions in `canonical_desk`:

```text
up     : glyph-local +Y -> canonical-desk +Y (0 deg)
right  : glyph-local +Y -> canonical-desk +X (-90 deg)
down   : glyph-local +Y -> canonical-desk -Y (180 deg)
left   : glyph-local +Y -> canonical-desk -X (90 deg)
```

For example:

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m sim.mujoco.replay.rule_based_multistage_demo \
  --dataset-root /path/to/dataset --episode 0 \
  --stage-a 0:120 --stage-b 250:400 \
  --fixed-trajectory-id digit_1_v1 \
  --fixed-trajectory-size-m 0.08 \
  --fixed-trajectory-direction right \
  --output /tmp/rule_based_digit_1_right --record-depth --no-sleep
```

`digit_1_v1` through `digit_9_v1` are nine independent one-stroke
centerlines. The first point of each path is anchored to the measured TCP
boundary after stage A; changing size or direction does not move that anchor.
The path remains in `canonical_desk`, and the existing
`DeskTcpPositionControl` performs the contract-checked desk-to-world mapping.
The default digit speed is the selected source replay spans' right-TCP average
speed, excluding the gap between spans. The resulting actual boundary is then
used by the existing relative stage-B rebase. A future `1`-through-`9`
sequence must be declared as nine such trajectory phases; it is not implicitly
treated as one continuous stroke.

```bash
PYTHONPATH=. python - <<'PY'
from sim.mujoco.replay.umi_dataset import load_umi_replay_episode
episode = load_umi_replay_episode("/tmp/rule_based_mujoco_demo", 0)
print(episode.action_dim, len(episode.frames))
PY
```
