# MuJoCo checkpoint 组件与接入进度

2026-10-24：已按用户要求恢复开发，并在原保存点上合入固定 development
`b48d23aac3f0`。本轮完成 native scene、acceleration planner 和 actuated owner
组件；完整 rig、Gateway 会话及 RPent 联合恢复仍待装配。现有入口不自动启用这些组件。

## 已保存的实现

`sim/mujoco/native_checkpoint.py` 是仅支持已验证 MuJoCo 3.10.0 的原生物理组件。
调用方必须已持有物理状态的排他边界；组件自身不暂停线程、不取得 Gateway admission
或控制 authority。恢复构造独立 `MjData`/model 候选，不修改运行中的 owner，
也不调用 reset、forward 或 step。

- 复用官方 native payload，保存编译模型、积分器和求解缓存，不另写全数组序列化器。
- 补充官方 3.10 codec 漏存的 9 个 scalar 和 3 个固定 vector：约束数量、awake
  有效长度、lazy-stage 标志及 solver 诊断。原生载荷与辅助状态共同参与校验。
- 检查版本、ABI、显式载荷大小限制、有限积分状态、模型身份与原生即时字节往返。
- 拒绝 process callback、plugin、活跃 stack/threadpool；这些检查不能代替调用方互斥。
- 记录源 compilation signatures。官方反序列化后的 signatures 为零，
  **不支持恢复 `MjSpec.bind/edit/recompile` 关系**。
- 仅处理可信来源的 native bytes；SHA 不提供来源认证，原生解析器也不是安全沙箱。
  `max_native_bytes` 限制序列化载荷，不等于反序列化总内存上限。

字段依据为 [MuJoCo 3.10 官方 codec](https://github.com/google-deepmind/mujoco/blob/3.10.0/python/mujoco/structs_wrappers.cc#L816-L975)
和 [原生字段定义](https://github.com/google-deepmind/mujoco/blob/3.10.0/include/mujoco/mjxmacro.h#L998-L1040)。

## 当前验证及其含义

在 Python 3.10.20 / MuJoCo 3.10.0 上执行：

```bash
bash scripts/run_pytest.sh tests/test_mujoco_native_checkpoint.py tests/test_robot_motion_trajectory_checkpoint.py tests/test_mujoco_actuation_checkpoint.py tests/test_sim_actuation_checkpoint_clock.py -q
```

本轮加上原 actuator、gripper、planner 与 execution rig 的受影响回归，共 **202 passed**。
新增模块与测试的 Ruff check/format 通过；旧 `scene.py` 和 `gripper.py` 原有的
11 条 lint 告警未增加，保留基线状态。本轮未做完整 profile、渲染、远端、模型或任务验收。

立即恢复要求未改写的 native bytes 完全一致；后续相同动作续跑比较所有已分配、
公开的语义字段及模型 hash，包含辅助状态和 contact/solver/warning/timer 成员。
墙钟 `timer.duration` 不作确定性判据；稠密求解下未使用的四个 sparse-J 索引不比较；
dual/island arena 按官方 codec 的实际启用条件选择，不能读取未分配指针。
按字段比较 contact 避免把 C 结构尾部 padding 当成物理差异，不改保存物或 live state。
原先失败及有限重复测试的证据保留，没有覆盖成最终通过结果。

## 新增的 owner 接口

| 层 | 入口 | 所保留的状态与边界 |
| --- | --- | --- |
| scene facade | `MujocoScene.from_native_checkpoint` | 在新 model/data 上重建 joint/body/geom/camera 索引和 metadata，不调用 forward/reset/step；projector/scratch 从新 scene 重新创建 |
| 位置 planner | `robot_motion.position_control.trajectory_checkpoint` | 完整 q/qd/qdd、目标及安全目标历史、滤波与制动状态、时间、generation/token；必填目标逻辑时钟，要求 command transaction 已结束 |
| arm/gripper wrapper | `sim.mujoco.actuation.checkpoint` | 固定 typed component，复用原 core 恢复；保留完整 safety projection、held command、缓存及 queued intent，不调用 wrapper reset，不改 native ctrl |

Planner 恢复使用调用方给定的 limits/configuration，必须与保存值一致，不从保存物放宽限制。
目标 q/sequence/publication 和 safe-target history 成组校验，未知字段与未完成 proposal 拒绝。
该 codec 留在 `position_control` 包内，rig 只消费其接口，不从诊断 snapshot 猜回状态。

SceneSpec 使用现有 typed `to_dict/from_dict` 重建；其只读映射不能整体 deepcopy。
实际双臂 CAD 模型已验证两侧手臂、两只夹爪及 `front/left_wrist/right_wrist` 绑定，
候选 native bytes/auxiliary 不变。元数据与完整 profile 身份的绑定仍属于后续装配责任。

Actuated owner 组件保存 actual native time，恢复时逐值核对，再检验 owner 的完成时刻。
不同 clock 的浮点累加与 tick×dt 误差按有界 ULP 处理，误差界必须小于半个 tick；
不调整任何时间或 deadline。这不是互斥证明，父 rig 仍须排除中途事务与并发接纳。
faulted owner、额外 guard/oracle、未知 owner 实现不在该首阶段资格内。

真实 CAD 用例在第 193 个 physics tick 保存非稳定状态：arm 有延迟命令、gripper
有非零速率和新 queued intent。新 scene 与新 owner 装配后，对相同后续输入逐 tick
比较了 40 个 tick 的 qpos/qvel/ctrl/act/warmstart、公开 snapshot 及 queued receipt。
Native 与 planner 各自有新解释器测试；完整 owner bundle 尚没有磁盘/跨进程验收。

既有 `reset(initial_state=...)` 可恢复 core pending，同时将新 wrapper 的本地 receipt
初始化为 `(0,None)`；该状态原样保存，不由 core 补造历史。若已有非零本地 receipt，
对应 pending 的 sequence/target 必须一致。完整 payload、来源及生命周期绑定由后续
profile/store 负责；本层不证明任意两个组件来自同一段历史。

## 修正的延迟队列时间窗

原 arm/gripper 恢复校验错误要求所有待处理事件都不早于“当前 tick 结束时刻”。
实际 step 在 tick 起点消费事件，例如起点 `0.009`、终点 `0.010`、事件 `0.0095`：
事件尚未消费，应在下一个 tick 处理，不能删除或提前执行。

- Arm 用现有整数 `physics_tick` 与物理周期恢复最后处理边界。
- Gripper 支持可变 dt，snapshot 增加 `last_step={physics_time_sec,timestep_sec}`，
  记录真实处理区间；恢复检查该区间的结束时刻与 resume time 一致。
- 已处理边界及其原消费 epsilon 内的事件仍拒绝。旧 v1 缺少 `last_step` 时维持
  原保守时间界限，不猜测历史。新 typed component 要求完整 core 往返一致。
- 公共 API 反例先复现 2 失败、1 通过；修复后覆盖区间内未处理事件、已处理事件拒绝、
  非 1ms 步长、损坏区间和旧格式保守行为。`hold_targets()` 与正常控制计算没有改变。

## 尚未完成，不能据此放开上层

1. Gateway admission、物理线程和控制 owner 的联合静止边界。
2. 将上述 owner 组件组合成完整 rig，补 mailbox/reference、序号、逻辑时钟和严格资格；
   再处理 target 的 authority、执行 gate、IK/safety 与保留引用。
3. 新实例中 FK、collision、payload、observer、scratch 等消费者的统一绑定。
4. profile 身份、持久化/原子提交、历史产物与恢复分支的来源关系。
5. sleep 混合状态、完整 no-model profile、渲染和远端组合验收。
6. RPent SDK/runtime/记录前缀与上述下层合同的联合恢复。

这些组件尚无生产消费者。Gateway native checkpoint 和未完成后台执行的既有拒绝不变。
后续先检查最新 development/correction 与并行改动，再继续 owner 合同；不得将当前
组件回归通过视为完整 profile 已实现，也不得用 fresh start/reset 冒充精确冷恢复。
