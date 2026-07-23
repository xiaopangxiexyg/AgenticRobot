# Sim Human-in-the-Loop（HIL）使用说明

本文档对应当前已经落地的 `CS-6b1f` 窄分支：在 MuJoCo 中让 replay 保持
初始 authority，被动 human pose source 通过 typed takeover 完成一次安全接管，
并由外部 episode recorder 形成可重放的 control trace。

当前 profile 是 **sim-only、default-off、library/test composition**，不是新的
production entrypoint，也不会改变既有 teleop、replay、deploy 或 RTC 路径。

## 1. 当前能力与边界

| 项目 | 当前语义 |
|---|---|
| 运行目标 | MuJoCo authoritative runtime |
| 初始 authority | replay |
| human 输入 | `PassiveTeleopPoseSessionSource`，只观察、不自动发布 |
| takeover | typed `request_human_takeover`，先 HOLD/cancel，再等待 fresh emission |
| readiness | 当前 sim 使用 `VirtualSimHumanTakeoverReadinessProvider`；只产生虚拟 deadman/rebase evidence |
| arm 表示 | HIL seam 按声明的 pose effectors 工作；当前 CS-6b1f target profile 是 `bimanual_ee_pose_v1`，target 内部才 pose→coupled IK→qpos |
| capture | 外部 `EpisodeTraceRecorderSink`；`hold_before_unbind` 后 WAL/seal/commit |
| 真机副作用 | 无。接入真实遥操设备时也只能作为 passive source，不能绕过 real deadman/rebase gate |
| production caller | 无。必须由显式测试/应用代码组装和启动 |

当前 HIL 没有第二套 queue、replay cursor、epoch、dispatcher 或 episode owner。
新的遥操设备、sim target 或模型 source 应优先复用已有 source/target/provider/emission
seam；不要按设备、checkpoint 或 simulator 新建 composition。

### 1.1 本体切换：通用 HIL seam 与当前 MuJoCo profile 分开

HIL 的 takeover、HOLD/cancel、readiness、emission promotion、publication barrier
和外部 recorder 都不携带“双臂”假设。`PassiveTeleopPoseSourceSpec` 现在按声明的
`reference_frame_by_effector` 与 `action_spec_hash_by_effector` 决定 pose effector
集合；默认值仍保留历史双臂 contract/hash，单臂只需建立一个新的数据化 source
contract，例如：

```python
from control_station.sources import PassiveTeleopPoseSourceSpec

human_spec = PassiveTeleopPoseSourceSpec.for_arm(
    "right_arm",
    reference_frame="right_arm_base",
    action_spec_hash="rm65-tool-tcp-pose-v1",
    source_id="human-single-arm",
    source_profile_id="single-arm-tracker-v1",
)
```

多个非双臂 effectors 也可通过 `for_effectors(...)` 显式声明。其余步骤仍复用同一
`TeleopSession(rig=None, recorder=None)`、`HumanTakeoverOperationSpec`、readiness
provider、promotion 和 recorder seam；不会为单臂新增 queue、authority、epoch 或
episode owner。spec 的 effector/frame/action-spec 集合进入 semantic hash，少一侧、
多一侧或错配都会在 source/target binding 阶段 fail closed。

当前 `CS-6b1f` MuJoCo target 是一个明确的 **RM65 coupled-bimanual capability
profile**，其 action catalog、pose→coupled-IK resolver 和 execution manifest 仍要求
`left_arm + right_arm`。因此把上面的单臂 spec 直接塞进该 profile 会被拒绝，这是
能力不兼容而不是 HIL seam 不通用。接入真正单臂本体时只需提供该本体自己的 target
catalog、pose/frame/IK（或其它 execution）adapter 与 capability/manifest evidence，
再复用同一 Station/HIL transaction；不得复制一套按单臂命名的 handoff/replay/record
实现。夹爪、6DOF/16DOF 灵巧手仍属于独立的 qpos/native component gate，不因 arm
cardinality 改成 pose。

## 2. 最小可运行验证（R1，无真实设备）

从仓库根目录、激活 `dexteleop310` 环境执行：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_control_station_mujoco_pose_human_replay.py::test_cs6b1f_sim_hil_promotes_passive_pose_and_keeps_publish_seam_unique
```

该测试验证：

1. replay 可以先成为唯一 authority；passive human poll 不产生 human proposal；
2. takeover worker 在 HOLD 下等待 takeover boundary 之后的 immutable emission；
3. virtual readiness 完成后，原 emission identity/content/timing 只 promotion/admit 一次；
4. human pose 经现有 safety、scheduled pose→qpos target 和 MuJoCo planner 执行；
5. 后续 passive emission 仍只通过已有 human publication seam 提交。

正式 capture/WAL/committed-reader 验证：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_control_station_mujoco_pose_human_replay.py::test_cs6b1f_formal_hil_capture_seals_and_rebuilds_pose_lineage
```

该测试会在 pytest 临时目录中创建 trace WAL、seal、commit 和 ActualDecisionView，
不会写入真实数据集，也不会连接设备。

## 3. 应用侧组装顺序

当前没有通用 CLI；应用代码应复用以下顺序，不要在 GUI 或 source 中另建控制循环：

```text
build MuJoCo runtime + pose target
    -> build replay source/binding
    -> build PassiveTeleopPoseSessionSource (optional)
    -> build_scoped_cs6b1f_mujoco_pose_human_replay_station(...)
    -> external EpisodeTraceRecorderSink.prepare/bind
    -> connect target, replay, human; run preflight
    -> request replay candidate + replay authority
    -> poll passive human pose (observe only)
    -> register HumanTakeoverOperationSpec + readiness provider
    -> request_human_takeover (worker performs HOLD/invalidation/authority transfer)
    -> poll one post-boundary pose emission
    -> wait for human_takeover_ready
    -> promote_human_takeover_emission(spec_id) exactly once
    -> step MuJoCo / record decisions
    -> release human authority, verify HOLD
    -> recorder.release_episode(..., hold_before_unbind)
    -> seal / commit / committed reader
```

真实 VIVE 或其它设备只替换 passive source 的构造部分。可复用现有
`teleop_runtime.configs.build_bimanual_passive_session(...)`，然后包装为
`PassiveTeleopPoseSessionSource`；该 session 必须保持 `rig=None`、`recorder=None`，
并在实际连接前完成设备侧只读检查。当前 real deadman/rebase typed receipt 尚未实现，
因此真实设备不能把 `VirtualSimHumanTakeoverReadinessProvider` 替代为 production READY。

## 4. 使用 `sim/mujoco_temp_pipeline` checkpoint

### 4.1 先做独立策略基线

临时 PPO pipeline 的默认 checkpoint 和 task config：

```bash
CHECKPOINT=sim/mujoco_temp_pipeline/checkpoints/latest.pt
TASK=sim/configs/tasks/rm65_right_cube_lift_rl_normal_v1.json

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 8 --num-envs 4 --device cpu
```

有图形环境时再运行 viewer：

```bash
PYTHONPATH=. python -m sim.mujoco_temp_pipeline.play \
  --checkpoint "$CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 1 --device cpu --print-safety-events
```

换 checkpoint 或 task config 时必须保持 checkpoint hash 兼容。loader 会在 hash 不一致时
fail closed；`--allow-task-config-mismatch` 只用于碰撞回归诊断，不能作为 HIL 或策略部署配置。
当前工作树中的历史 `cube_lift_normal_v1.pt` 可能由旧 task JSON 生成，若上面的 evaluate
报告 mismatch，不要使用 override 冒充有效策略。先生成与当前 task config 匹配的短 smoke
checkpoint：

```bash
SMOKE_CHECKPOINT=/tmp/dexteleop_hil_cube_lift_normal_current.pt
SMOKE_RECORD_ROOT=/tmp/dexteleop_hil_temp_records

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.train \
  --task-config "$TASK" \
  --checkpoint "$SMOKE_CHECKPOINT" \
  --record-root "$SMOKE_RECORD_ROOT" \
  --run-name hil_smoke \
  --num-envs 1 --rollout-steps 8 --minibatch-size 8 \
  --update-epochs 1 --total-timesteps 16 --device cpu

PYTHONPATH=. python -m sim.mujoco_temp_pipeline.evaluate \
  --checkpoint "$SMOKE_CHECKPOINT" \
  --task-config "$TASK" \
  --episodes 1 --num-envs 1 --device cpu
```

该 smoke checkpoint 只验证当前 task/action/normalizer/safety 链路，不代表策略质量；
正式策略仍需足够训练步数和独立成功率验收。

`privileged_expert` checkpoint 只能作为 simulator/debug teacher，不能进入真实或通用
policy source。

### 4.2 当前动作契约与 CS-6b1f 的不兼容点

| temp pipeline 输出 | CS-6b1f 要求 |
|---|---|
| 7-D：右臂 6 维 normalized joint-target delta + 右夹爪 close delta | 双臂 `bimanual_ee_pose_v1` arm action |
| 右臂控制，左臂固定 home | 左右 arm 同一 coupled selection group |
| qpos/native gripper action | arm source/selected/safe pose；hand 另有 qpos component gate |
| `CubeLiftEnv` 自己的 observation/safety/task loop | 已启动的 authoritative MuJoCo runtime + immutable emission |
| normal 15-D observation；privileged 31-D observation | 目标 runtime 的已声明 observation contract |

因此 checkpoint **不能直接**传给
`build_scoped_cs6b1f_mujoco_pose_human_replay_station()`，也不能把 7-D qpos delta
伪装成 pose emission。当前推荐顺序是：

1. 用 `evaluate/play` 验证 checkpoint 的任务成功率、collision projection、rejection 和 tracking error；
2. 在不取得 authority、不发送动作的 shadow adapter 中加载同一 checkpoint 和 normalizer，记录 observation/action/hash；
3. 只有完成显式 observation contract、delta→target 解码、right-arm FK→pose、gripper component contract、safety 和 downstream admission receipt 后，才考虑接入 policy source；
4. policy→human 或 policy→replay 的正式混合另属于后续 independent policy-source gate，不自动并入 CS-6b1f。

这条路径仍然复用 `PolicyActionEmission` / `PolicyDeployRuntime` 的 lower-owned
emission seam；模型侧不需要为 RTC 或 Station 增加专用 adapter。需要的只是应用侧
observation/action representation adapter，并且 adapter 必须保留 checkpoint schema、
task-config hash、normalizer 和 source lineage。

## 5. 进入下一阶段前的验收

- `evaluate` 使用的 checkpoint/task hash 一致；normal policy 而非 privileged expert；
- shadow adapter 输出 finite、shape 与 contract 一致，且不调用 `submit_*` 或 authority API；
- 每个 policy output 有唯一 immutable emission identity、source validation 和 timing；
- qpos delta 不直接进入当前 arm mixed source；转换为 pose 后才能进入 pose source seam；
- hand/gripper qpos 只有在 active-mask、joint order、units、bounds、safety 和 downstream receipt 都冻结后才可 compound；
- 默认关闭时，既有 `sync_queue_v1`、legacy replay、teleop、deploy 和 RTC 测试结果不变。

## 6. 相关入口

- [Control Station README](../../control_station/README.md)
- [Control Station architecture visual guide](../../todos/dexteleop_architecture_visual_guide.md)
- [MuJoCo temporary PPO pipeline](../../sim/mujoco_temp_pipeline/README.md)
- [MuJoCo simulation README](../sim/README.md)
